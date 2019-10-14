"""
Run few-shot learning on FashionProductImaes dataset using code from github
epo https://github.com/oscarknagg/few-shotOmniglot reproducing results of
Snell et al Prototypical Networks.
"""
from torchvision import models
import argparse
from few_shot_learning.train_few_shot import few_shot_training

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
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--n_val_classes', default=10, type=int)
args = parser.parse_args()

n_epochs = 1
validation_episodes = 200
evaluation_episodes = 1000
episodes_per_epoch = 100


few_shot_training(
    dataset=args.dataset,
    num_input_channels=3,
    drop_lr_every=20,
    validation_episodes=validation_episodes,
    evaluation_episodes=evaluation_episodes,
    episodes_per_epoch=episodes_per_epoch,
    n_epochs=n_epochs,
    small_dataset=args.small_dataset,
    n_train=args.n_train,
    n_test=args.n_test,
    k_train=args.k_train,
    k_test=args.k_test,
    q_train=args.q_train,
    q_test=args.q_test,
    distance=args.distance,
    pretrained=args.pretrained,
    monitor_validation=args.validate,
    n_val_classes=args.n_val_classes,
    architecture='resnet18'
)