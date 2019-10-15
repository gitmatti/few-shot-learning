import argparse
import torchvision.models as models
import torch.nn.parallel

from few_shot_learning.train_transfer import transfer
from config import DATA_PATH


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Transfer Learning')
parser.add_argument('--data', metavar='DIR',
                    default=DATA_PATH,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCHITECTURE',
                    default='resnet18',
                    choices=model_names,
                    help=('model architecture: '
                          + ' | '.join(model_names)
                          + ' (default: resnet18)'))
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_tr', '--learning-rate-tr', default=None,
                    type=float, metavar='LR',
                    help='initial learning rate for transfer',
                    dest='lr_tr')
parser.add_argument('--optim', default=torch.optim.Adam,
                    metavar='OPTIMIZER',
                    help='optimizer from torch.optim')
parser.add_argument('--optim-args', default={}, type=dict, metavar='DICT',
                    help='optimizer args')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--distributed', action='store_true',
                    help='Use multi-processing distributed training to '
                         'launch N processes per node, which has N GPUs.')
parser.add_argument('--date', action='store_true',
                    help='Create log and model folder with current date')
parser.add_argument('--small-dataset', action='store_true',
                    help='Use dataset with smaller image size')
# TODO model_dir and log_dir

args = parser.parse_args()

transfer(
    data_dir=args.data,
    architecture=args.arch,
    num_workers=args.workers,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
    learning_rate_tr=args.lr_tr,
    optimizer_cls=args.optim,
    print_freq=args.print_freq,
    seed=args.seed,
    gpu=args.gpu,
    distributed=args.distributed,
    date_prefix=args.date,
    small_dataset=args.small_dataset
)