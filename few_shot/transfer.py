import argparse
import os
import random
import time
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from few_shot.datasets import FashionProductImages
from few_shot.utils import AverageMeter, ProgressMeter, accuracy,\
    save_checkpoint, save_results

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Transfer Learning')
parser.add_argument('--data', metavar='DIR',
                    default=os.path.expanduser("~/data"),
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCHITECTURE', default='resnet18',
                    choices=model_names,
                    help=('model architecture: '
                          + ' | '.join(model_names)
                          + ' (default: resnet18)'))
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--optim', default=torch.optim.Adam, metavar='OPTIMIZER',
                    help='optimizer from torch.optim')
parser.add_argument('--optim-args', default={}, type=dict, metavar='DICT',
                    help='optimizer args')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
# TODO not elegant
parser.add_argument('--device', default=None, metavar='DEV')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        args.device = torch.device('cuda', args.gpu)
    else:
        if torch.cuda.is_available():
            args.device = torch.device('cuda', 0)
        else:
            args.device = torch.device('cpu')

    # ngpus = torch.cuda.device_count()

    global best_acc1

    # ----------------------------------------------------------------------- #
    # Data loading
    # ----------------------------------------------------------------------- #
    datadir = args.data

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((80, 60), scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((80, 60)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((80, 60)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    # prepare dictionary of all datasets. There are datasets for train, test,
    # and validation for both the top20 and bottom (transfer) classes
    data = {
        classes: {
            split: FashionProductImages(
                datadir,
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

    train_loader = torch.utils.data.DataLoader(
        trainset_ft, batch_size=args.batch_size, num_workers=args.workers,
        sampler=train_sampler_ft, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        valset_ft, batch_size=args.batch_size, num_workers=args.workers,
        sampler=val_sampler_ft, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset_ft, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )

    # ----------------------------------------------------------------------- #
    # Create model and optimizer for initial fine-tuning
    # ----------------------------------------------------------------------- #
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    # TODO: optimizer args
    optimizer_ft = args.optim(model.parameters(), args.lr)

    # TODo test this
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_ft.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # change the last layer of the pre-trained model for fine-tuning
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, trainset_ft.n_classes)

    model.to(args.device)
    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)

    # TODO args via argparse
    lr_scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft,
                                                      step_size=5,
                                                      gamma=0.5)

    # TODO validate for both fine-tuning and transfer
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # ----------------------------------------------------------------------- #
    # Training: fine-tune
    # ----------------------------------------------------------------------- #
    print("=> Running {} epochs of fine-tuning (top20)".format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        # TODO scheduler
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer_ft, lr_scheduler_ft,
              epoch, args)

        # evaluate on validation set
        top1, _ = validate(val_loader, model, criterion, args)
        acc1 = top1.avg

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer_ft.state_dict(),
        }, is_best)

    test_top1_ft, test_top5_ft = validate(test_loader, model, criterion, args)

    # TODO not implemented
    save_results({})

    # ----------------------------------------------------------------------- #
    # Create model and optimizer for transfer learning
    # ----------------------------------------------------------------------- #
    # ending _tr for transfer
    trainset_tr = data['bottom']['train']
    valset_tr = data['bottom']['val']
    testset_tr = data['bottom']['test']

    # can't stratify along classes since some have only one sample
    # TODO: make sure there is at least one sample of every class in the
    # TODO: training set?
    train_sampler_tr, train_indices_tr, val_sampler_tr, val_indices_tr = \
        get_train_and_val_sampler(trainset_tr, balanced_training=True,
                                  stratify=False)

    train_loader_tr = torch.utils.data.DataLoader(
        trainset_tr, batch_size=args.batch_size, num_workers=args.workers,
        sampler=train_sampler_tr
    )
    val_loader_tr = torch.utils.data.DataLoader(
        valset_tr, batch_size=args.batch_size, num_workers=args.workers,
        sampler=val_sampler_tr
    )
    test_loader_tr = torch.utils.data.DataLoader(
        testset_tr, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True
    )

    # modify output layer a second time
    model.fc = nn.Linear(model.fc.in_features, trainset_tr.n_classes)
    model.to(args.device)

    # freeze all lower layers of the network
    # for param in model_tr.parameters():
    #     param.requires_grad = False

    # TODO: different learning rate for transfer in argparse
    # TODO optimizer args
    optimizer_tr = args.optim(model.parameters(), lr=1e-4)

    lr_scheduler_tr = torch.optim.lr_scheduler.StepLR(optimizer_tr,
                                                      step_size=10,
                                                      gamma=0.1)

    # ----------------------------------------------------------------------- #
    # Training: transfer
    # ----------------------------------------------------------------------- #
    print("=> Running {} epochs of transfer learning".format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader_tr, model, criterion, optimizer_tr, lr_scheduler_tr,
              epoch, args)

        # evaluate on validation set
        top1, _ = validate(val_loader_tr, model, criterion, args)
        acc1 = top1.avg

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer_ft.state_dict(),
        }, is_best)

    test_top1_tr, test_top5_tr = validate(test_loader_tr,
                                          model, criterion, args)

    # TODO not implemented
    save_results({})


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],  # [batch_time, data_time]
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        images.to(args.device)
        target.to(args.device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    scheduler.step()


def validate(val_loader, model, criterion, args):
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
            images.to(args.device)
            target.to(args.device)

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

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1, top5


def get_train_and_val_sampler(trainset, train_size=0.9, balanced_training=True,
                              stratify=True):
    n_classes = trainset.n_classes
    n_samples = len(trainset)
    indices = np.arange(n_samples)
    labels = trainset.target_indices

    train_indices, val_indices = train_test_split(
        indices, train_size=train_size,
        stratify=trainset.target_indices if stratify else None
    )

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    if balanced_training:
        class_sample_count = np.bincount(labels[train_indices],
                                         minlength=n_classes)
        class_sample_count = torch.from_numpy(class_sample_count).float()

        class_weights = torch.zeros_like(class_sample_count)
        class_weights[class_sample_count > 0] = \
            1. / class_sample_count[class_sample_count > 0]

        train_weights = class_weights[labels[train_indices]]
        dataset_weights = torch.zeros(n_samples)
        dataset_weights[train_indices] = train_weights

        # TODO: in this way, train_loader will still produce n_samples
        # samples per epoch (instead of train_size*n_samples)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            dataset_weights, n_samples, replacement=True)
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    return train_sampler, train_indices, val_sampler, val_indices


if __name__ == '__main__':
    main()