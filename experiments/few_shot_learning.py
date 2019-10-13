"""
Run few-shot learning on FashionProductImaes dataset using code from github
epo https://github.com/oscarknagg/few-shotOmniglot reproducing results of
Snell et al Prototypical Networks.
"""
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
import argparse
import numpy as np

from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs

from few_shot_learning.datasets import FashionProductImages,\
    FashionProductImagesSmall
from few_shot_learning.models import Identity
from config import DATA_PATH, PATH


setup_dirs()
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=30, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--small-dataset', action='store_true')
parser.add_argument('-a', '--arch', metavar='ARCHITECTURE',
                    default='resnet18',
                    choices=model_names,
                    help=('model architecture: '
                          + ' | '.join(model_names)
                          + ' (default: resnet18)'))
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'fashion':
    n_epochs = 40
    dataset_class = FashionProductImagesSmall if args.small_dataset else FashionProductImages
    num_input_channels = 3
    drop_lr_every = 20
else:
    raise(ValueError, 'Unsupported dataset')

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_small={args.small_dataset}'

print(param_str)

###################
# Create datasets #
###################
resize = (80, 60) if args.small_dataset else (400, 300)

background_transform = transforms.Compose([
    transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
    # transforms.RandomGrayscale(),
    transforms.RandomPerspective(),
    transforms.RandomHorizontalFlip(),
    # transforms.Resize(resize),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])

background = dataset_class(DATA_PATH, split='all', classes='background',
                           transform=background_transform)
background_sampler = NShotTaskSampler(background, episodes_per_epoch,
                                      args.n_train, args.k_train, args.q_train)
background_taskloader = DataLoader(
    background,
    batch_sampler=background_sampler,
    num_workers=4
)

evaluation_transform = transforms.Compose([
    transforms.Resize(resize),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
])

evaluation = dataset_class(DATA_PATH, split='all', classes='evaluation',
                           transform=background_transform)
evaluation_sampler = NShotTaskSampler(evaluation, episodes_per_epoch,
                                      args.n_test, args.k_test, args.q_test)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=evaluation_sampler,
    num_workers=4
)


#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
# model = models.__dict__[args.arch](pretrained=True)
# model.fc = Identity()
model.to(device, dtype=torch.double)


############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().to(device)


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train,
                         'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)
