import unittest

import torch
from torch.utils.data import random_split, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms

from few_shot_learning.datasets import FashionProductImages,\
    FashionProductImagesSmall

from config import DATA_PATH

TOP20 = ["Jeans", "Perfume and Body Mist", "Formal Shoes",
         "Socks", "Backpacks", "Belts", "Briefs",
         "Sandals", "Flip Flops", "Wallets", "Sunglasses",
         "Heels", "Handbags", "Tops", "Kurtas",
         "Sports Shoes", "Watches", "Casual Shoes", "Shirts",
         "Tshirts"]

CLS = (FashionProductImagesSmall, FashionProductImages)

SIZE = 44441


class TestFashionProductImages(unittest.TestCase):
    def test_default_config(self):
        for cls in CLS:
            # TODO actually test this
            if DATA_PATH is None:
                self.assertRaises(Exception, cls, DATA_PATH)
            else:
                dataset = cls(DATA_PATH)

    def test_split_arg(self):
        wrong_arg = "split"
        for cls in CLS:
            self.assertRaises(AssertionError, cls, DATA_PATH, split=wrong_arg)

    def test_top_bottom_classes(self):
        for cls in CLS:
            for split in ['test', 'train']:
                # create top and bottom dataset
                top = cls(DATA_PATH, split=split, classes='top')
                bottom = cls(DATA_PATH, split=split, classes='bottom')

                self.assertEqual(top.n_classes, 20,
                                 "Attribute n_classes should be 20 for "
                                 + "classes='top'")

                self.assertTrue(top.samples[top.target_type].isin(TOP20).all(),
                                "There are samples not in the top20 classes "
                                "for classes='top'")
                # self.assertTrue(dataset_top.targets.isin(TOP20).all())

                self.assertFalse(
                    bottom.samples[bottom.target_type].isin(TOP20).any(),
                    "There are samples in the top20 classes for "
                    "classes='bottom'")

    def test_train_test_set(self):
        for cls in CLS:
            for classes in ['top', 'bottom']:
                train = cls(DATA_PATH, split="train", classes=classes)
                test = cls(DATA_PATH, split="test", classes=classes)

                self.assertTrue((train.samples["year"] % 2 == 0).all(),
                                "Training set contains samples from odd years")
                self.assertFalse((test.samples["year"] % 2 == 0).any(),
                                 "Test set contains samples from even years")

    def test_size(self):
        for cls in CLS:
            total_size = 0
            for split in ["train", "test"]:
                for classes in ["top", "bottom"]:
                    dataset = cls(DATA_PATH, split=split, classes=classes)
                    total_size += len(dataset)
            self.assertTrue(total_size == SIZE,
                            "The size of all splits/class configs does not sum"
                            "to {} for {}".format(SIZE, cls))

            total_size = 0
            for split in ["train", "test"]:
                dataset = cls(DATA_PATH, split=split, classes=None)
                total_size += len(dataset)
            self.assertTrue(total_size == SIZE,
                            "The size of all splits for all classes does not"
                            "sum to {} for {}".format(SIZE, cls))

    def _test_indices(self):
        for cls in CLS:
            for split in ["train", "test"]:
                dataset = cls(DATA_PATH, split=split, classes=None)
                for i in range(len(dataset)):
                    _, _ = dataset[i]

    def _test_image_format_and_transforms(self):
        for cls in CLS:
            transform1 = transforms.Compose([
                transforms.Resize((80, 60)),
            ])
            dataset = cls(DATA_PATH, classes=None, transform=transform1)

            all_correctly_sized = True
            all_rgb = True
            for i in range(len(dataset)):
                X, _ = dataset[i]
                all_correctly_sized &= X.size == (60, 80)
                all_rgb &= X.mode == 'RGB'
            self.assertTrue(all_correctly_sized)
            self.assertTrue(all_rgb)

            transform2 = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset = cls(DATA_PATH, classes=None, transform=transform2)

            all_torch_tensors = True
            all_chw_format = True
            for i in range(len(dataset)):
                X, _ = dataset[i]
                all_torch_tensors &= isinstance(X, torch.Tensor)
                all_chw_format &= len(X.size) == 3
            self.assertTrue(all_torch_tensors)
            self.assertTrue(all_chw_format)


class TestDataloaderOnFashionProductImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = FashionProductImages(DATA_PATH, split='train', classes=None)

    # TODO figure out of any of this is necessary
    def test_random_split(self):
        train_size = int(len(self.dataset) * 0.8)

        trainset, valset = random_split(
            self.dataset,[train_size, len(self.dataset) - train_size])

        self.assertTrue(len(trainset) + len(valset) == len(self.dataset))
