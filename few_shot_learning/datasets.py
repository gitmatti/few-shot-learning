from __future__ import print_function
import os
import csv
import numpy as np
from torchvision.datasets import VisionDataset
from functools import partial
import PIL.Image
import pandas
from sklearn.preprocessing import LabelEncoder
import zipfile

from config import DATA_PATH


class FashionProductImages(VisionDataset):
    """TODO
    """
    base_folder = 'fashion-dataset'
    filename = "fashion-product-images-dataset.zip"
    url = None # TODO
    file_list = None # TODO

    top20_classes = [
        "Jeans", "Perfume and Body Mist", "Formal Shoes",
        "Socks", "Backpacks", "Belts", "Briefs",
        "Sandals", "Flip Flops", "Wallets", "Sunglasses",
        "Heels", "Handbags", "Tops", "Kurtas",
        "Sports Shoes", "Watches", "Casual Shoes", "Shirts",
        "Tshirts"]

    background_classes = [
        "Cufflinks", "Rompers", "Laptop Bag", "Sports Sandals", "Hair Colour",
        "Suspenders", "Trousers", "Kajal and Eyeliner", "Compact", "Concealer",
        "Jackets", "Mufflers", "Backpacks", "Sandals", "Shorts", "Waistcoat",
        "Watches", "Pendant", "Basketballs", "Bath Robe", "Boxers",
        "Deodorant", "Rain Jacket", "Necklace and Chains", "Ring",
        "Formal Shoes", "Nail Polish", "Baby Dolls", "Lip Liner", "Bangle",
        "Tshirts", "Flats", "Stockings", "Skirts", "Mobile Pouch", "Capris",
        "Dupatta", "Lip Gloss", "Patiala", "Handbags", "Leggings", "Ties",
        "Flip Flops", "Rucksacks", "Jeggings", "Nightdress", "Waist Pouch",
        "Tops", "Dresses", "Water Bottle", "Camisoles", "Heels", "Gloves",
        "Duffel Bag", "Swimwear", "Booties", "Kurtis", "Belts",
        "Accessory Gift Set", "Bra"
    ]

    evaluation_classes = [
        "Jeans", "Bracelet", "Eyeshadow", "Sweaters", "Sarees", "Earrings",
        "Casual Shoes", "Tracksuits", "Clutches", "Socks", "Innerwear Vests",
        "Night suits", "Salwar", "Stoles", "Face Moisturisers",
        "Perfume and Body Mist", "Lounge Shorts", "Scarves", "Briefs",
        "Jumpsuit", "Wallets", "Foundation and Primer", "Sports Shoes",
        "Highlighter and Blush", "Sunscreen", "Shoe Accessories",
        "Track Pants", "Fragrance Gift Set", "Shirts", "Sweatshirts",
        "Mask and Peel", "Jewellery Set", "Face Wash and Cleanser",
        "Messenger Bag", "Free Gifts", "Kurtas", "Mascara", "Lounge Pants",
        "Caps", "Lip Care", "Trunk", "Tunics", "Kurta Sets", "Sunglasses",
        "Lipstick", "Churidar", "Travel Accessory"
    ]
    
    # TODO.not_implemented: should different 'target_type' be allowed?
    target_type = 'articleType'

    def __init__(self, root=DATA_PATH, split='train', transform=None,
                 target_transform=None, download=False, classes='top'):
        super(FashionProductImages, self).__init__(
            root, transform=transform, target_transform=target_transform)

        # self.transform = transforms.Compose([
        #    transforms.Resize((80, 60)),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #])

        assert split in ['train', 'test', 'all']
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')            

        fn = partial(os.path.join, self.root, self.base_folder)

        # TODO.refactor: refer to class attribute 'file_list' instead of "styles.csv"
        with open(fn("styles.csv")) as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)

        # additional column for comma artifacts in column 'productDisplayName'
        column_names.append(column_names[-1] + '2')

        # TODO.refactor: clean up column names, potentially merge last two columns
        self.df_meta = pandas.read_csv(fn("styles.csv"), names=column_names,
                                       skiprows=1)

        # introduce column with image filenames
        self.df_meta = self.df_meta.assign(
            filename=self.df_meta["id"].apply(lambda x: str(x) + ".jpg"))
        
        # relevant classes either by 'top'/'bottom'/'background'/'evaluation'
        # keyword or by list
        all_classes = set(self.df_meta[self.target_type])
        if classes is not None:
            if isinstance(classes, list):
                assert set(classes).issubset(all_classes)
            else:
                assert classes in ['top', 'bottom', 'background', 'evaluation']
                if classes == 'top':
                    classes = self.top20_classes
                elif classes == 'bottom':
                    classes = list(all_classes.difference(self.top20_classes))
                elif classes == 'background':
                    classes = self.background_classes
                else:
                    classes = self.evaluation_classes
        else:
            classes = list(all_classes)
        
        # parses out samples that
        # - have a the relevant class label
        # - have an image present in the 'images' folder
        # - confer to the given split 'train'/'test'/'all'
        images = os.listdir(fn("images"))
        if self.split == 'train':
            split_mask = self._train_mask(self.df_meta)
        elif self.split == 'test':
            split_mask = ~ self._train_mask(self.df_meta)
        else:
            split_mask = True
        
        self.samples = self.df_meta[
            (self.df_meta[self.target_type].isin(classes))
            & (self.df_meta["filename"].isin(images))
            & split_mask
        ]

        self.targets = self.df[self.target_type]
        self.target_codec = LabelEncoder()
        self.target_codec.fit(classes)

        self.target_indices = self.target_codec.transform(self.targets)
        self.n_classes = len(self.target_codec.classes_)
        self.classes = self.target_codec.classes_

        # assign different columns to integrate with few-shot github repo
        self.samples = self.samples.assign(my_id=self.samples["id"])
        self.samples = self.samples.assign(id=np.arange(len(self.samples)))
        self.samples = self.samples.assign(
            class_id=self.target_codec.transform(
                self.samples[self.target_type]
            )
        )

    def __getitem__(self, index):
        sample = self.samples["filename"].iloc[index]
        X = PIL.Image.open(
            os.path.join(
                self.root,
                self.base_folder,
                "images",
                sample
            )
        ).convert("RGB")
        target = self.target_indices[index]

        # TODO.extension: allow returning one-hot representation of target

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.samples)
    
    def _train_mask(self, df):
        return df["year"] % 2 == 0

    # to integrate with few-shot github repo
    @property
    def num_classes(self):
        return self.n_classes

    # to integrate with few-shot github repo
    @property
    def df(self):
        df = {

        }
        return self.samples

    def download(self):
        raise NotImplementedError

    def _check_integrity(self):
        return True


class FashionProductImagesSmall(FashionProductImages):
    """TODO
    """
    base_folder = 'fashion-product-images-small'
    url = None # TODO
    filename = 'fashion-product-images-small.zip'


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((80, 60)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((80, 60)),
            transforms.ToTensor(),
        ]),
    }

    fashion_data = FashionProductImagesSmall(
        "~/data", transform=data_transforms["train"])

    train_size = int(len(fashion_data) * 0.75)
    trainset, valset = random_split(
        fashion_data, [train_size, len(fashion_data) - train_size])

    train_loader = DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    counter = 0
    for batch in train_loader:
        X, y = batch
        counter += X.shape[0]

    assert counter == train_size

    counter = 0
    for batch in val_loader:
        X, y = batch
        counter += X.shape[0]

    assert counter == len(fashion_data) - train_size


