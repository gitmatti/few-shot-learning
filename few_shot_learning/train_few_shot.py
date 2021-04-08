"""
Run few-shot learning on FashionProductImaes dataset using code from github
repo https://github.com/oscarknagg/few-shot under

MIT License

Copyright (c) 2019 Oscar Knagg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

reproducing results of
Snell et al Prototypical Networks. In places where substantial changes have
been made to the original code, this is marked with an ADAPTED/BEFORE comment
"""
import torch
from torch.optim import Adam
import torch.nn.parallel
from torch.utils.data import DataLoader
from torchvision import transforms, models
import warnings
import numpy as np
from typing import Callable, Tuple

from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, create_nshot_task_label
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from few_shot.metrics import categorical_accuracy

from few_shot_learning.datasets import FashionProductImages, \
    FashionProductImagesSmall
from few_shot_learning.models import Identity
from config import DATA_PATH, PATH


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def few_shot_training(
        datadir=DATA_PATH,
        dataset='fashion',
        num_input_channels=3,
        drop_lr_every=20,
        validation_episodes=200,
        evaluation_episodes=1000,
        episodes_per_epoch=100,
        n_epochs=80,
        small_dataset=False,
        n_train=1,
        n_test=1,
        k_train=30,
        k_test=5,
        q_train=5,
        q_test=1,
        distance='l2',
        pretrained=False,
        monitor_validation=False,
        n_val_classes=10,
        architecture='resnet18',
        gpu=None
):
    setup_dirs()

    if dataset == 'fashion':
        dataset_class = FashionProductImagesSmall if small_dataset \
            else FashionProductImages
    else:
        raise (ValueError, 'Unsupported dataset')

    param_str = f'{dataset}_nt={n_train}_kt={k_train}_qt={q_train}_' \
                f'nv={n_test}_kv={k_test}_qv={q_test}_small={small_dataset}_' \
                f'pretrained={pretrained}_validate={monitor_validation}'

    print(param_str)

    ###################
    # Create datasets #
    ###################

    # ADAPTED: data transforms including augmentation
    resize = (80, 60) if small_dataset else (400, 300)

    background_transform = transforms.Compose([
        transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
        # transforms.RandomGrayscale(),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        # transforms.Resize(resize),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize(resize),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

    if monitor_validation:
        if not n_val_classes >= k_test:
            n_val_classes = k_test
            print("Warning: `n_val_classes` < `k_test`. Take a larger number"
                  " of validation classes next time. Increased to `k_test`"
                  " classes")

        # class structure for background (training), validation (validation),
        # evaluation (test): take a random subset of background classes
        validation_classes = list(
            np.random.choice(dataset_class.background_classes, n_val_classes))
        background_classes = list(set(dataset_class.background_classes).difference(
            set(validation_classes)))

        # use keyword for evaluation classes
        evaluation_classes = 'evaluation'

        # Meta-validation set
        validation = dataset_class(datadir, split='all',
                                   classes=validation_classes,
                                   transform=evaluation_transform)
        # ADAPTED: in the original code, `episodes_per_epoch` was provided to
        # `NShotTaskSampler` instead of `validation_episodes`.
        validation_sampler = NShotTaskSampler(validation, validation_episodes,
                                              n_test, k_test, q_test)
        validation_taskloader = DataLoader(
            validation,
            batch_sampler=validation_sampler,
            num_workers=4
        )
    else:
        # use keyword for both background and evaluation classes
        background_classes = 'background'
        evaluation_classes = 'evaluation'

    # Meta-training set
    background = dataset_class(datadir, split='all',
                               classes=background_classes,
                               transform=background_transform)
    background_sampler = NShotTaskSampler(background, episodes_per_epoch,
                                          n_train, k_train, q_train)
    background_taskloader = DataLoader(
        background,
        batch_sampler=background_sampler,
        num_workers=4
    )

    # Meta-test set
    evaluation = dataset_class(datadir, split='all',
                               classes=evaluation_classes,
                               transform=evaluation_transform)
    # ADAPTED: in the original code, `episodes_per_epoch` was provided to
    # `NShotTaskSampler` instead of `evaluation_episodes`.
    evaluation_sampler = NShotTaskSampler(evaluation, evaluation_episodes,
                                          n_test, k_test, q_test)
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=evaluation_sampler,
        num_workers=4
    )

    #########
    # Model #
    #########
    
    if torch.cuda.is_available():
        if gpu is not None:
            device = torch.device('cuda', gpu)
        else:
            device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    
    if not pretrained:
        model = get_few_shot_encoder(num_input_channels)
        # ADAPTED
        model.to(device)
        # BEFORE
        # model.to(device, dtype=torch.double)
    else:
        assert torch.cuda.is_available()
        model = models.__dict__[architecture](pretrained=True)
        model.fc = Identity()
        if gpu is not None:
            model = model.cuda(gpu)
        else:
            model = model.cuda()
        # TODO this is too risky: I'm not sure that this can work, since in
        #  the few-shot github repo the batch axis is actually split into
        #  support and query samples
        # model = torch.nn.DataParallel(model).cuda()

    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr

    ############
    # Training #
    ############
    print(f'Training Prototypical network on {dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().to(device)

    callbacks = [
        # ADAPTED: this is the test monitoring now - and is only done at the
        # end of training.
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,  # THIS IS NOT USED
            n_shot=n_test,
            k_way=k_test,
            q_queries=q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(n_test, k_test, q_test, device=device),
            distance=distance,
            on_epoch_end=False,
            on_train_end=True,
            prefix='test_'
        )
    ]
    if monitor_validation:
        callbacks.append(
            # ADAPTED: this is the validation monitoring now - computed
            # after every epoch.
            EvaluateFewShot(
                eval_fn=proto_net_episode,
                num_tasks=evaluation_episodes,  # THIS IS NOT USED
                n_shot=n_test,
                k_way=k_test,
                q_queries=q_test,
                # BEFORE taskloader=evaluation_taskloader,
                taskloader=validation_taskloader,  # ADAPTED
                prepare_batch=prepare_nshot_task(n_test, k_test, q_test, device=device),
                distance=distance,
                on_epoch_end=True,  # ADAPTED
                on_train_end=False,  # ADAPTED
                prefix='val_'
            )
        )
    callbacks.extend([
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_{n_test}-shot_{k_test}-way_acc',
            verbose=1,  # ADAPTED
            save_best_only=monitor_validation  # ADAPTED
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
    ])

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(n_train, k_train, q_train, device=device),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': n_train, 'k_way': k_train,
                             'q_queries': q_train, 'train': True,
                             'distance': distance},
    )


# ADAPTED: the original code used torch.double
def prepare_nshot_task(n: int, k: int, q: int, device=None) -> Callable:
    """Typical n-shot task preprocessing.

    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """

    def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Create 0-k label and move to GPU.

        TODO: Move to arbitrary device
        """
        x, y = batch
        # BEFROE x = x.double().cuda()
        x = x.to(device)  # ADPATED
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k, q).to(device)
        return x, y

    return prepare_nshot_task_


class EvaluateFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 on_epoch_end: bool = True,
                 on_train_end: bool = False,
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

        # ADAPTED
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    # ADAPTED
    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end:
            self._validate(epoch, logs=logs)

    # ADAPTED
    def on_train_end(self, epoch, logs=None):
        if self._on_train_end:
            self._validate(epoch, logs=logs)

    # ADAPTED
    def _validate(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * \
                                        y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved
    with the epoch number and the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less

        # BEFORE: THIS IS A BUG
        # self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s'
                                % (epoch + 1, self.monitor, self.best,
                                   current, filepath))
                        self.best = current
                        torch.save(self.model.state_dict(), filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: %s did not improve from %0.5f' %
                                (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (
                    epoch + 1, filepath))
                torch.save(self.model.state_dict(), filepath)
