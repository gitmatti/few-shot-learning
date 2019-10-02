from __future__ import print_function
import os
import csv
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from functools import partial
import PIL
import pandas
from sklearn.preprocessing import LabelEncoder
import zipfile


class FashionProductImages(VisionDataset):
    """TODO
    """
    base_folder = 'fashion-product-images-small'  # 'fashion-dataset'
    # base_folder_small_dataset = 'fashion-product-images-small'
    url = None # TODO
    filename = "fashion-product-images-small.zip"  # 'fashion-product-images-dataset.zip'
    file_list = None # TODO

    top20_classes = [
        "Jeans", "Perfume and Body Mist", "Formal Shoes",
        "Socks", "Backpacks", "Belts", "Briefs",
        "Sandals", "Flip Flops", "Wallets", "Sunglasses",
        "Heels", "Handbags", "Tops", "Kurtas",
        "Sports Shoes", "Watches", "Casual Shoes", "Shirts",
        "Tshirts"]

    def __init__(self, root, split="train", target_type="articleType",
                 transform=None, target_transform=None, download=False,
                 small_dataset=False):
        super(FashionProductImages, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.split = split
        self.target_type = target_type

        # TODO.not_implemented: should a list of target types be allowed?
        self.target_type = target_type

        # TODO.not_implemented: allow for usage of small dataset
        # if small_dataset:
        #    base_folder = self.base_folder_small_dataset
        # else:
        #    base_folder = self.base_folder

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

        column_names.append(column_names[-1] + '2')

        # TODO.refactor: clean up column names, potentially merge last two columns
        # TODO.refactor: column year is not parsed as integer - why?
        self.df_meta = pandas.read_csv(fn("styles.csv"), names=column_names,
                                       skiprows=1)

        # checks if the images from 'styles.csv' are actually present
        # in the 'images' folder and parses out top20 classes
        images = os.listdir(fn("images"))
        self.samples = self.df_meta.loc[
            (self.df_meta[self.target_type].isin(self.top20_classes))
            & (self.df_meta["id"].apply(lambda x: str(x) + ".jpg").isin(
                images))
            ]

        self.targets = self.samples[self.target_type]
        self.target_codec = LabelEncoder()
        self.target_codec.fit(self.targets)

        # TODO.decision: are additional codecs necessary?
        # self.article_codec = LabelEncoder()
        # self.gender_codec = LabelEncoder()
        # self.master_category_codec = LabelEncoder()
        # self.season_codec = LabelEncoder()
        # self.article_codec.fit(self.metadata.loc[:, "articleType"])

        # TODO.decision: prepare indices here or when __getitem__ is called?
        self.target_indices = self.target_codec.transform(self.targets)

        self.n_classes = len(self.target_codec.classes_)

        # TODO.goal: test and training set master split
        # TODO.goal: fine-tune vs transfer classes

    def __getitem__(self, index):
        # TODO.check: some images are not RGB?
        # TODO.check: some images are not 80x60?
        sample = str(self.samples["id"].iloc[index]) + ".jpg"
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images",
                                        sample)).convert("RGB")
        target = self.target_indices[index]

        # TODO.extension: allow returning one-hot representation of target

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        # TODO.not_implemented: check this and compare to e.g. MNIST/CIFAR
        # TODO.not_implemented: how to download from Kaggle
        # if self._check_integrity():
        #    print('Files already downloaded and verified')
        #    return

        # for (file_id, md5, filename) in self.file_list:
        #    download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        # with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
        #    f.extractall(os.path.join(self.root, self.base_folder))

        raise NotImplementedError

    def _check_integrity(self):
        # TODO.not_implemented: check this and compare to e.g. MNIST/CIFAR

        # for (_, md5, filename) in self.file_list:
        #    fpath = os.path.join(self.root, self.base_folder, filename)
        #    _, ext = os.path.splitext(filename)
        #    # Allow original archive to be deleted (zip and 7z)
        #    # Only need the extracted images
        #    if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
        #        return False

        # Should check a hash of the images
        # return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))
        return True

    def extra_repr(self):
        # TODO.not_implemented
        pass


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

    fashion_data = FashionProductImages(
        "~/data", transform=data_transforms["train"])

    train_size = int(len(fashion_data) * 0.75)
    trainset, valset = random_split(
        fashion_data, [train_size, len(fashion_data) - train_size])

    train_loader = DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=4)

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


