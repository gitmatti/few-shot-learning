"""
Run transfer learning on FashionProductImages dataset. This will first fine-tune
a chosen model from the ImageNet model zoo on classifying the 20 most common
product and then in a second pass will fine-tune the network further on the
remaining, less common, product classes.
"""
import os
import random
import time
import numpy as np
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.parallel

from few_shot_learning.models import AdaptiveHeadClassifier
from few_shot_learning.datasets import FashionProductImages, \
    FashionProductImagesSmall
from few_shot_learning.sampler import get_train_and_val_sampler
from few_shot_learning.utils import AverageMeter, ProgressMeter, \
    allocate_model, accuracy, allocate_inputs, save_checkpoint, save_results
from config import DATA_PATH

best_acc1 = 0


def transfer(
        data_dir=DATA_PATH,
        architecture='resnet18',
        num_workers=4,
        epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        learning_rate_tr=None,
        optimizer_cls=torch.optim.Adam,
        print_freq=10,
        seed=None,
        gpu=None,
        dtype=None,
        distributed=False,
        log_dir='~/few-shot-learning/logs',
        model_dir='~/few-shot-learning/models',
        date_prefix=False,
        small_dataset=False
):
    log_dir = os.path.expanduser(log_dir)
    model_dir = os.path.expanduser(model_dir)
    if date_prefix:
        date = datetime.now().strftime(r"%y_%m_%d_%H%M")
        log_dir = os.path.join(log_dir, date)
        model_dir = os.path.join(model_dir, date)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # make partial functions for allocation of model and inputs
    _allocate_model = partial(allocate_model, dtype, distributed, gpu,
                              architecture)

    _allocate_inputs = partial(allocate_inputs, dtype, distributed, gpu)

    # TODO not_implemented
    # optionally resume from a checkpoint
    # if resume:
    #    restore_model(model, optimizer, gpu, model_dir)

    # TODO not_implemented
    # if evaluate:
    #    validate(val_loader, model, criterion, device, print_freq)
    #    return

    # ----------------------------------------------------------------------- #
    # Data loading
    # ----------------------------------------------------------------------- #

    # Imagenet-specific normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # image dimension resize depending on dataset
    resize = (80, 60) if small_dataset else (400, 300)

    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(resize),
            transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            normalize
        ]),
    }

    # prepare dictionary of all datasets. There are datasets for train, test,
    # and validation for both the top20 and bottom (transfer) classes
    dataset = FashionProductImages if not small_dataset \
        else FashionProductImagesSmall

    data = {
        classes: {
            split: dataset(
                data_dir,
                split='train' if split in ['train', 'val'] else 'test',
                classes=classes,
                transform=data_transforms[split]
            ) for split in ["train", "test", "val"]
        } for classes in ["top", "bottom"]
    }

    # ending _ft is for initial fine-tuning with top20 classes
    trainset_ft = data['top']['train']
    valset_ft = data['top']['val']
    testset_ft = data['top']['test']

    # train and val sampler
    train_sampler_ft, train_indices_ft, val_sampler_ft, val_indices_ft = \
        get_train_and_val_sampler(trainset_ft,
                                  train_size=0.9,
                                  balanced_training=True)

    train_loader_ft = torch.utils.data.DataLoader(
        trainset_ft, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler_ft, pin_memory=True
    )

    val_loader_ft = torch.utils.data.DataLoader(
        valset_ft, batch_size=batch_size, num_workers=num_workers,
        sampler=val_sampler_ft, pin_memory=True
    )

    test_loader_ft = torch.utils.data.DataLoader(
        testset_ft, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    # ending _tr for transfer
    trainset_tr = data['bottom']['train']
    valset_tr = data['bottom']['val']
    testset_tr = data['bottom']['test']

    # can't stratify along classes since some have only one sample
    # TODO: make sure there is at least one sample of every class in the
    #  training set?
    train_sampler_tr, train_indices_tr, val_sampler_tr, val_indices_tr = \
        get_train_and_val_sampler(trainset_tr, balanced_training=True,
                                  stratify=False)

    train_loader_tr = torch.utils.data.DataLoader(
        trainset_tr, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler_tr
    )
    val_loader_tr = torch.utils.data.DataLoader(
        valset_tr, batch_size=batch_size, num_workers=num_workers,
        sampler=val_sampler_tr
    )
    test_loader_tr = torch.utils.data.DataLoader(
        testset_tr, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    # ----------------------------------------------------------------------- #
    # Create model and optimizer for initial fine-tuning
    # ----------------------------------------------------------------------- #
    print("=> using pre-trained model '{}'".format(architecture))
    out_features = [trainset_ft.n_classes, trainset_tr.n_classes]
    model = AdaptiveHeadClassifier(out_features, architecture=architecture)
    model = _allocate_model(model)

    # define loss function (criterion) and optimizer
    # TODO: different devices
    criterion = nn.CrossEntropyLoss().cuda()

    # TODO: optimizer args
    optimizer_ft = optimizer_cls(model.parameters(), learning_rate)

    # TODO parameters as function arguments
    lr_scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft,
                                                      step_size=5,
                                                      gamma=0.7)

    # ----------------------------------------------------------------------- #
    # Training: fine-tune
    # ----------------------------------------------------------------------- #
    print("=> Running {} epochs of fine-tuning (top20)".format(epochs))

    train_model(train_loader_ft, val_loader_ft, model, criterion, optimizer_ft,
                lr_scheduler_ft, epochs, print_freq, _allocate_inputs,
                model_prefix="finetuning",
                log_dir=os.path.expanduser(log_dir),
                model_dir=os.path.expanduser(model_dir))

    _, test_top1_ft, test_top5_ft = validate(test_loader_ft, model, criterion,
                                             print_freq, _allocate_inputs)

    # ----------------------------------------------------------------------- #
    # Create optimizer for transfer learning
    # ----------------------------------------------------------------------- #

    # change the active head
    try:
        model.set_active(1)
    except AttributeError:
        model.module.set_active(1)

    # start a new learning rate scheduler and optimizer
    learning_rate_tr = learning_rate if learning_rate_tr is None \
        else learning_rate_tr
    optimizer_tr = optimizer_cls(model.parameters(), lr=learning_rate_tr)

    lr_scheduler_tr = torch.optim.lr_scheduler.StepLR(optimizer_tr,
                                                      step_size=5,
                                                      gamma=0.7)

    # ----------------------------------------------------------------------- #
    # Training: transfer
    # ----------------------------------------------------------------------- #
    print("=> Running {} epochs of transfer learning".format(epochs))

    train_model(train_loader_tr, val_loader_tr, model, criterion, optimizer_tr,
                lr_scheduler_tr, epochs, print_freq, _allocate_inputs,
                model_prefix="transfer",
                log_dir=os.path.expanduser(log_dir),
                model_dir=os.path.expanduser(model_dir))

    _, test_top1_tr, test_top5_tr = validate(test_loader_tr,
                                             model, criterion,
                                             print_freq, _allocate_inputs)


def train_model(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        epochs,
        print_freq,
        allocate_inputs,
        model_prefix,
        log_dir,
        model_dir
):
    monitor_variables = ("train_loss", "train_acc1", "train_acc5",
                         "val_loss", "val_acc1", "val_acc5")

    results = {v: np.zeros(epochs) for v in monitor_variables}

    best_acc1 = 0.0
    best_state_dict = None

    for epoch in range(epochs):

        # train for one epoch
        train_loss, train_top1, train_top5 = train_epoch(train_loader, model,
                                                         criterion, optimizer,
                                                         lr_scheduler, epoch,
                                                         print_freq,
                                                         allocate_inputs)

        # evaluate on validation set
        val_loss, top1, top5 = validate(val_loader, model, criterion,
                                        print_freq, allocate_inputs)

        # remember best acc@1 and save checkpoint
        acc1 = top1.avg
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            best_state_dict = model.state_dict()

        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_dir=model_dir, prefix=model_prefix)

        for key, result in zip(monitor_variables,
                               (train_loss, train_top1, train_top5,
                                val_loss, top1, top5)):
            results[key][epoch] = result.avg

    model.load_state_dict(best_state_dict)
    save_results(results, dir=log_dir, prefix=model_prefix)


def train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch,
                print_freq, allocate_inputs):
    # monitoring progress
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5, batch_time, data_time],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    since = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - since)

        images, target = allocate_inputs(images, target)

        # compute output and loss
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - since)
        since = time.time()

        if i % print_freq == 0:
            progress.display(i)

    # anneal learning rate
    scheduler.step()

    return losses, top1, top5


def validate(val_loader, model, criterion, print_freq, allocate_inputs):
    # batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],  # [batch_time]
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = allocate_inputs(images, target)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses, top1, top5