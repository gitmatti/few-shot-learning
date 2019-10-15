import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from few_shot.models import get_few_shot_encoder
from few_shot.proto import proto_net_episode
from few_shot.metrics import categorical_accuracy
from few_shot.core import NShotTaskSampler
from typing import Callable, Tuple

from few_shot_learning.train_few_shot import prepare_nshot_task
from few_shot_learning.models import Identity, AdaptiveHeadClassifier
from few_shot_learning.datasets import FashionProductImagesSmall,\
    FashionProductImages
from few_shot_learning.utils import accuracy
from config import DATA_PATH

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def evaluate_transfer(
        state_dict,
        device,
        architecture='resnet50',
        classes='top',
        small_dataset=False,
        distributed=False,
        batch_size=128,
        num_workers=4,
):
    resize = (80, 60) if small_dataset else (400, 300)

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset_class = FashionProductImagesSmall if small_dataset \
        else FashionProductImages

    testset = dataset_class(DATA_PATH, classes=classes, split="test",
                            transform=transform)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    model = models.__dict__[architecture](pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, testset.n_classes)
    if distributed:
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model.eval()

    outputs = []
    targets = []

    with torch.no_grad():
        # end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            # compute outputallocate_inputs_(images, target)
            output = model(images)
            # loss = criterion(output, target)

            outputs.append(output)
            targets.append(target)

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    acc1_avg = acc1.item()
    acc5_avg = acc5.item()

    # print(acc1_avg, acc5_avg)

    acc1_classwise = np.zeros(
        testset.n_classes)  # [0.0 for _ in range(testset.n_classes)]
    acc5_classwise = np.zeros(
        testset.n_classes)  # [0.0 for _ in range(testset.n_classes)]

    for c in range(testset.n_classes):
        is_class = (targets == c)
        if is_class.any():
            acc1, acc5 = accuracy(outputs[is_class, :], targets[is_class],
                                  topk=(1, 5))
            acc1_classwise[c] = acc1.item()
            acc5_classwise[c] = acc5.item()
        else:
            acc1_classwise[c] = -1.0
            acc5_classwise[c] = -1.0

    return acc1_avg, acc5_avg, acc1_classwise, acc5_classwise


def evaluate_few_shot(
        state_dict,
        n_shot,
        k_way,
        q_queries,
        device,
        architecture='resnet18',
        pretrained=False,
        small_dataset=False,
        metric_name=None,
        evaluation_episodes=1000,
        num_input_channels=3,
        distance='l2'
):
    if not pretrained:
        model = get_few_shot_encoder(num_input_channels)
        model.load_state_dict(state_dict)
    else:
        # assert torch.cuda.is_available()
        model = models.__dict__[architecture](pretrained=True)
        model.fc = Identity()
        model.load_state_dict(state_dict)

    dataset_class = FashionProductImagesSmall if small_dataset \
        else FashionProductImages

    # Meta-test set
    resize = (80, 60) if small_dataset else (400, 300)
    evaluation_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    evaluation = FashionProductImagesSmall(
        DATA_PATH, split='all', classes='evaluation',
        transform=evaluation_transform
    )
    sampler = NShotTaskSampler(evaluation, evaluation_episodes,
                               n_shot, k_way, q_queries)
    taskloader = DataLoader(
        evaluation,
        batch_sampler=sampler,
        num_workers=4
    )
    prepare_batch = prepare_nshot_task(n_shot, k_way, q_queries)

    if metric_name is None:
        metric_name = f'test_{n_shot}-shot_{k_way}-way_acc'
    seen = 0
    totals = {'loss': 0, metric_name: 0}

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().to(device)

    for batch_index, batch in enumerate(taskloader):
        x, y = prepare_batch(batch)

        loss, y_pred = proto_net_episode(
            model,
            optimiser,
            loss_fn,
            x,
            y,
            n_shot=n_shot,
            k_way=k_way,
            q_queries=q_queries,
            train=False,
            distance=distance
        )

        seen += y_pred.shape[0]

        totals['loss'] += loss.item() * y_pred.shape[0]
        totals[metric_name] += categorical_accuracy(y, y_pred) * \
                               y_pred.shape[0]

    totals['loss'] = totals['loss'] / seen
    totals[metric_name] = totals[metric_name] / seen

    return totals


class ToSize(object):
    """Return height and width for  a ``PIL Image`` or ``numpy.ndarray``

    Converts a PIL Image or numpy.ndarray (H x W x C) to a tuple (H, W).
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image for which size is to be
            returned.

        Returns:
            Tuple: Image size.
        """
        if isinstance(pic, PIL.Image.Image):
            return pic.size
        elif isinstance(pic, numpy.ndarray):
            return pic.shape[:2]
        else:
            raise TypeError

    def __repr__(self):
        return self.__class__.__name__ + '()'
