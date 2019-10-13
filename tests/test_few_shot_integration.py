import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import Module
from typing import Callable
import numpy as np

from few_shot.core import NShotTaskSampler, prepare_nshot_task
# from few_shot.models import Flatten

from few_shot_learning.datasets import FashionProductImages, FashionProductImagesSmall
from config import DATA_PATH




def dummy_fit_function(model: Module,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       train: bool):
    """Performs a single dummy training episode behaving as e.g. a
    proto_net_episode
    """
    if train:
        # Zero gradients
        model.train()
    else:
        model.eval()

    # Embed all samples
    embeddings = model(x)
    embeddings = embeddings.view(embeddings.size(0), -1)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]

    # Dummy-Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries*k_way, k_way) = (num_queries, k_way)
    distances = torch.randn(q_queries * k_way, k_way)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    return loss, y_pred


num_input_channels = 3


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dummy_model = nn.Sequential(
    nn.Conv2d(num_input_channels, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
).to(device, dtype=torch.double)


class TestFashionFewShotIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # syntax of github repo few-shot calls meta-training-set = background
        # and meta-test-set = evaluation
        cls.transform = transforms.Compose([
            transforms.Resize((80, 60)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        cls.background = FashionProductImages(DATA_PATH, split='all',
                                              classes='background',
                                              transform=cls.transform)
        cls.evaluation = FashionProductImages(DATA_PATH, split='all',
                                              classes='evaluation',
                                              transform=cls.transform)
        cls.n_background_classes = cls.background.n_classes

        cls.background_class_count = np.bincount(
            cls.background.target_indices, minlength=cls.background.n_classes)

    def test_wrong_k_way_arg(self):
        episodes_per_epoch = 100
        n_shot = 5
        # make k-way larger than number of classes
        k_way = self.n_background_classes + 1
        q_queries = 1

        sampler = NShotTaskSampler(self.background,
                                   episodes_per_epoch,
                                   n_shot, k_way, q_queries)

        def dummy_iter():
            for sample in sampler:
                continue

        self.assertRaises(ValueError, dummy_iter)

    def test_wrong_q_query_arg(self):
        episodes_per_epoch = 1
        n_shot = 5
        k_way = 5

        # taking q_query larger than the largest class
        # should immediately raise an error
        q_queries = self.background_class_count.max() + 1

        sampler = NShotTaskSampler(self.background,
                                   episodes_per_epoch,
                                   n_shot, k_way, q_queries)

        def dummy_iter():
            for sample in sampler:
                continue

        self.assertRaises(ValueError, dummy_iter)

        # taking q_query larger than the smallest class should raise an error
        # as soon as that class is selected in an episode
        q_queries = self.background_class_count.min() + 1
        class_min_index = self.background_class_count.argmin()

        sampler = NShotTaskSampler(self.background,
                                   episodes_per_epoch,
                                   n_shot, k_way, q_queries,
                                   fixed_tasks=[[class_min_index]])

        def dummy_iter():
            for sample in sampler:
                continue

        self.assertRaises(ValueError, dummy_iter)

    def test_background_sampler(self):
        episodes_per_epoch = 10
        n_train = 5
        k_train = 15
        q_train = 1

        background_sampler = NShotTaskSampler(self.background,
                                              episodes_per_epoch,
                                              n_train, k_train, q_train)

        background_loader = DataLoader(
            self.background,
            batch_sampler=background_sampler,
            num_workers=4
        )

        prepare_batch = prepare_nshot_task(n_train, k_train, q_train)

        for batch_index, batch in enumerate(background_loader):
            x, y = prepare_batch(batch)

            loss, y_pred = dummy_fit_function(
                dummy_model,
                torch.nn.NLLLoss(),
                x,
                y,
                n_shot=n_train,
                k_way=k_train,
                q_queries=q_train,
                train=False,
            )

    def test_evaluation_sampler(self):
        episodes_per_epoch = 10
        n_test = 1
        k_test = 5
        q_test = 1

        evaluation_sampler = NShotTaskSampler(self.evaluation,
                                              episodes_per_epoch,
                                              n_test, k_test, q_test)

        evaluation_loader = DataLoader(
            self.evaluation,
            batch_sampler=evaluation_sampler,
            num_workers=4
        )

        prepare_batch = prepare_nshot_task(n_test, k_test, q_test)

        for batch_index, batch in enumerate(evaluation_loader):
            x, y = prepare_batch(batch)

            loss, y_pred = dummy_fit_function(
                dummy_model,
                torch.nn.NLLLoss(),
                x,
                y,
                n_shot=n_test,
                k_way=k_test,
                q_queries=q_test,
                train=False,
            )
