import argparse
import os
import random
import time
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from few_shot.datasets import FashionProductImages
from few_shot.utils import AverageMeter, ProgressMeter, accuracy,\
    batchnorm_to_fp32, save_checkpoint, save_results

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
parser.add_argument('--distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs.')
# TODO not elegant
parser.add_argument('--device', default=None, metavar='DEV')
parser.add_argument('--dtype', default=None, metavar='DTYPE')

best_acc1 = 0


def main(
    datadir='~/data',
    architecture='resnet18',
    num_workers=4,
    epochs=100,
    start_epoch=0,
    batch_size=64,
    learning_rate=1e-3,
    optimizer=torch.optim.Adam,
    print_freq=10,
    resume=False,
    evaluate=False,
    seed=None,
    gpu=None,
    device=None,
    dtype=None,
    distributed=False
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    def _allocate_model(model):
        if dtype is not None:
            model = model.to(dtype)  
        
        if not distributed:
            if gpu is not None:
                device = torch.device('cuda', gpu)
            else:
                if torch.cuda.is_available() and not distributed:
                    device = torch.device('cuda', 0)
                else:
                    device = torch.device('cpu')
                    
            return model.to(device)
        else:
            # ngpus = torch.cuda.device_count()
            
            if architecture.startswith('alexnet') or architecture.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
          
            return model
        
    def _allocate_inputs(inputs, targets):
        if distributed:
            inputs = inputs.to(dtype).cuda()
            targets = targets.cuda()
        else:
            if gpu is not None:
                device = torch.device('cuda', gpu)
            else:
                if torch.cuda.is_available() and not distributed:
                    device = torch.device('cuda', 0)
                else:
                    device = torch.device('cpu')
            inputs = inputs.to(dtype).to(device)
            targets = targets.to(device)
        return inputs, targets
        
    # allocate_model = partial(_allocate_model, gpu, device, distributed self.base_folder)

    global best_acc1

    # ----------------------------------------------------------------------- #
    # Data loading
    # ----------------------------------------------------------------------- #
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((400,300)),
            # transforms.RandomResizedCrop((80, 60), scale=(0.8, 1.0)),
            # transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((400,300)),
            # transforms.Resize((80, 60)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((400,300)),
            # transforms.Resize((80, 60)),
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
        trainset_ft, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler_ft, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        valset_ft, batch_size=batch_size, num_workers=num_workers,
        sampler=val_sampler_ft, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset_ft, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True
    )

    # ----------------------------------------------------------------------- #
    # Create model and optimizer for initial fine-tuning
    # ----------------------------------------------------------------------- #
    print("=> using pre-trained model '{}'".format(architecture))
    model = models.__dict__[architecture](pretrained=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # TODO: optimizer args
    optimizer_ft = optimizer(model.parameters(), learning_rate)

    # TODo test this
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_ft.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # change the last layer of the pre-trained model for fine-tuning
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, trainset_ft.n_classes)
    model = _allocate_model(model)
    # model = batchnorm_to_fp32(model)

    # TODO parameters as function arguments
    lr_scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft,
                                                      step_size=5,
                                                      gamma=0.5)

    # TODO validate for both fine-tuning and transfer
    if evaluate:
        validate(val_loader, model, criterion, device, print_freq)
        return

    # ----------------------------------------------------------------------- #
    # Training: fine-tune
    # ----------------------------------------------------------------------- #
    print("=> Running {} epochs of fine-tuning (top20)".format(epochs))
    for epoch in range(start_epoch, epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer_ft, lr_scheduler_ft,
              epoch, print_freq, _allocate_inputs)

        # evaluate on validation set
        top1, _ = validate(val_loader, model, criterion, print_freq,
                           _allocate_inputs)
        acc1 = top1.avg

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer_ft.state_dict(),
        }, is_best)

    test_top1_ft, test_top5_ft = validate(test_loader, model, criterion,
                                          print_freq, _allocate_inputs)

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

    # modify output layer a second time
    model.fc = nn.Linear(model.fc.in_features, trainset_tr.n_classes)
    model = _allocate_model(model)
    # model.to(device)

    # freeze all lower layers of the network
    # for param in model_tr.parameters():
    #     param.requires_grad = False

    # TODO: different learning rate for transfer in argparse
    # TODO optimizer params as function arguments
    optimizer_tr = optimizer(model.parameters(), lr=learning_rate)

    lr_scheduler_tr = torch.optim.lr_scheduler.StepLR(optimizer_tr,
                                                      step_size=10,
                                                      gamma=0.1)

    # ----------------------------------------------------------------------- #
    # Training: transfer
    # ----------------------------------------------------------------------- #
    print("=> Running {} epochs of transfer learning".format(epochs))
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        train(train_loader_tr, model, criterion, optimizer_tr, lr_scheduler_tr,
              epoch, print_freq, _allocate_inputs)

        # evaluate on validation set
        top1, _ = validate(val_loader_tr, model, criterion, print_freq,
                           _allocate_inputs)
        acc1 = top1.avg

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer_ft.state_dict(),
        }, is_best)

    test_top1_tr, test_top5_tr = validate(test_loader_tr,
                                          model, criterion,
                                          print_freq, _allocate_inputs)

    # TODO not implemented
    save_results({})


def train(train_loader, model, criterion, optimizer,
          scheduler, epoch, print_freq, allocate_inputs):
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
        
        # compute output
        output = model(images)

        # TODO: CrossEntropyLoss accepts only torch.float32 !?   
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

    scheduler.step()


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
    args = parser.parse_args()
    main(
        data=args.data,
        architecture=args.arch,
        num_workers=args.workers,
        epochs=args.epoch,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optim,
        print_freq=args.print_freq,
        resume=args.resume,
        evaluate=args.evaluate,
        seed=args.seed,
        gpu=args.gpu,
        device=args.device,
        dtype=args.dtype,
        distributed=args.distributed
    )