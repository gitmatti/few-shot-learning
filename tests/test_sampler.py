import unittest

import torch
import pandas
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from few_shot_learning.sampler import get_train_and_val_sampler


class DummyUnbalancedDataset(Dataset):
    def __init__(self, n_classes=100, n_features=1, max_samples_per_class=100,
                 unstrafifiable=False):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.n_classes = n_classes
        self.n_features = n_features
        self.max_samples_per_class = max_samples_per_class
        self.samples_per_class = np.random.randint(0,
                                                   self.max_samples_per_class,
                                                   size=self.n_classes)
        self.samples_per_class[0] = 0
        if unstrafifiable:
            self.samples_per_class[1] = 1

        # Create a dataframe to be consistent with other Datasets
        self.target_indices = []
        for i in range(self.n_classes):
            self.target_indices.extend([i] * self.samples_per_class[i])

        self.target_indices = np.array(self.target_indices)

    def __len__(self):
        return self.samples_per_class.sum()

    def __getitem__(self, item):
        X = np.random.normal(size=self.n_features)
        y = self.target_indices[item]
        return X, y


class TestTrainAndValSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyUnbalancedDataset(max_samples_per_class=10000)
        cls.unstrafifiable_dataset = \
            DummyUnbalancedDataset(unstrafifiable=True)

    def test_stratify_indices(self):
        train_sampler, train_indices, val_sampler, val_indices = \
            get_train_and_val_sampler(self.dataset,
                                      train_size=0.6,
                                      balanced_training=False,
                                      stratify=True)

        # test proportion of labels according to selected indices first
        y_train_counts = np.bincount(
            self.dataset.target_indices[train_indices],
            minlength=self.dataset.n_classes)
        y_val_counts = np.bincount(
            self.dataset.target_indices[val_indices],
            minlength=self.dataset.n_classes)

        y_train_proportions = y_train_counts / y_train_counts.sum()
        y_val_proportions = y_val_counts / y_val_counts.sum()

        all_close = np.allclose(y_train_proportions, y_val_proportions,
                                rtol=1e-2, atol=1e-2)

        self.assertTrue(all_close,
                        "Label proportions in indices for training and "
                        "validation set are not close according to "
                        "np.allclose")

    def test_stratify_sampling(self):
        train_sampler, train_indices, val_sampler, val_indices = \
            get_train_and_val_sampler(self.dataset,
                                      train_size=0.6,
                                      balanced_training=False,
                                      stratify=True)

        # test proportion of labels when sampling
        train_loader = \
            DataLoader(self.dataset, batch_size=128, sampler=train_sampler)
        val_loader = \
            DataLoader(self.dataset, batch_size=128, sampler=val_sampler)

        y_train_counts = np.zeros(self.dataset.n_classes)
        y_val_counts = np.zeros(self.dataset.n_classes)
        for _, y in train_loader:
            y_train_counts += np.bincount(y, minlength=self.dataset.n_classes)
        for _, y in val_loader:
            y_val_counts += np.bincount(y, minlength=self.dataset.n_classes)

        y_train_proportions = y_train_counts / y_train_counts.sum()
        y_val_proportions = y_val_counts / y_val_counts.sum()

        all_close = np.allclose(y_train_proportions, y_val_proportions,
                                rtol=1e-2, atol=1e-2)

        self.assertTrue(all_close,
                        "Label proportions during sampling for training and "
                        "validation set are not close according to "
                        "np.allclose")

    def test_unstratifiable(self):
        self.assertRaises(Exception, get_train_and_val_sampler,
                          self.unstrafifiable_dataset, stratify=True)

    def test_train_size(self):
        self.assertRaises(ValueError, get_train_and_val_sampler, self.dataset,
                          train_size=1.1)
        self.assertRaises(ValueError, get_train_and_val_sampler, self.dataset,
                          train_size=1.1, stratify=False)

    def test_balanced_sampling(self):
        train_sampler, _, _, _ = \
            get_train_and_val_sampler(self.dataset,
                                      train_size=0.7,
                                      balanced_training=True,
                                      stratify=True)

        train_loader = \
            DataLoader(self.dataset, batch_size=128, sampler=train_sampler)

        y_counts = np.zeros(self.dataset.n_classes)
        for _, y in train_loader:
            y_counts += np.bincount(y, minlength=self.dataset.n_classes)

        y_proportions = y_counts / y_counts.sum()
        y_equi = np.ones(self.dataset.n_classes) / self.dataset.n_classes

        self.assertTrue(y_proportions[0] == 0.0)

        all_close = np.allclose(y_proportions[y_proportions > 0],
                                y_equi[y_proportions > 0],
                                rtol=1e-2, atol=1/(self.dataset.n_classes*10))

        self.assertTrue(all_close,
                        "Label proportions during sampling for training set "
                        "not close to uniform for balanced training according "
                        "to np.allclose")

    def test_no_overlap(self):
        pass

    def test_all_classes_in_trainset(self):
        pass

