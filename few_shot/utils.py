import shutil
import os
import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified
    values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def allocate_model(dtype, distributed, gpu, architecture, model):
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
        if architecture.startswith('alexnet') or architecture.startswith(
                'vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

        return model


def allocate_inputs(dtype, distributed, gpu, inputs, targets):
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


def batchnorm_to_fp32(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_fp32(child)
    return module


def restore_model(
        model,
        optimizer=None,
        gpu=None,
        model_dir='~/few-shot-learning/models',
        filename='model_best.pth.tar'
):
    model_filename = os.path.join(os.path.expanduser(model_dir), filename)

    if os.path.isfile(model_filename):
        print("=> loading checkpoint '{}'".format(model_filename))
        if gpu is None:
            checkpoint = torch.load(model_filename)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(model_filename, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(gpu)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_filename))


def save_checkpoint(state, is_best, prefix=None, filename='checkpoint.pth.tar',
                    dir="."):
    if prefix is not None:
        filename = '{}_{}'.format(prefix, filename)
        best_filename = '{}_{}'.format(prefix, 'model_best.pth.tar')
    else:
        best_filename = 'model_best.pth.tar'

    torch.save(state, os.path.join(dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir, filename),
                        os.path.join(dir, best_filename))


def save_results(results, prefix=None, filename='training_log.json', dir='.'):
    if prefix is not None:
        filename = '{}_{}'.format(prefix, filename)
    torch.save(results, os.path.join(dir, filename))

