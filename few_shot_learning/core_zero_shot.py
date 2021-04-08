"""
Core zero-shot learning functions for FashionProductImages dataset adapted from
github repo https://github.com/oscarknagg/few-shot under

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


The core methods
of the repo don't allow for zero-shot learning off-the-bat, but can be adapted
with minor changes.
"""
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

import numpy as np
from typing import List, Iterable, Callable, Tuple, Union

from few_shot.callbacks import *
from few_shot.metrics import categorical_accuracy
from few_shot.utils import pairwise_distances, mkdir
from few_shot.train import batch_metrics
from few_shot.core import create_nshot_task_label

from config import PATH


class ZeroShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 ):
        """PyTorch Sampler subclass that generates batches of zero-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(ZeroShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.q = q
        self.fixed_tasks = None  # fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(
                        self.dataset.df['class_id'].unique(), size=self.k,
                        replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[
                        self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[
                    self.dataset.df['class_id'].isin(episode_classes)]

                for k in episode_classes:
                    query = df[df['class_id'] == k].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)


def proto_net_zero_shot_episode(model: Module,
                                class_model: Module,
                                optimiser: Optimizer,
                                loss_fn: Callable,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                attr: torch.Tensor,
                                k_way: int,
                                q_queries: int,
                                distance: str,
                                train: bool):
    """TODO
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples and classes
    image_embeddings = model(x)
    class_embeddings = class_model(y, attr)

    assert image_embeddings.shape[1] == class_embeddings.shape[1]

    # Samples ZeroShotWrapper are straightforward as follows:
    # k lots of q query samples from those classes
    # support = class_embeddings
    queries = image_embeddings  # [n_shot*k_way:]
    # take every q_queries'th row
    prototypes = class_embeddings[np.arange(0, k_way * q_queries, q_queries)]

    # normalize prototypes to unit length
    prototype_norm = torch.norm(prototypes, p=2, dim=1, keepdim=True).detach()
    prototypes = prototypes.div(prototype_norm)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def prepare_zero_shot_task(k: int, q: int) -> Callable:
    """TODO
    """
    def prepare_zero_shot_task_(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO
        """
        x, y, attr = batch
        attr = attr.float()
        # x = x.double()# .cuda()
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k, q) #.cuda()
        return x, y, attr

    return prepare_zero_shot_task_


def setup_dirs():
    """Creates directories for this project."""
    mkdir(PATH + '/logs/')
    mkdir(PATH + '/logs/proto_nets')
    mkdir(PATH + '/logs/proto_nets/zero_shot')
    mkdir(PATH + '/models/')
    mkdir(PATH + '/models/proto_nets')
    mkdir(PATH + '/models/proto_nets/zero_shot')


class EvaluateZeroShot(Callback):
    """TODO
    """
    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 on_epoch_end: bool = True,
                 on_train_end: bool = False,
                 **kwargs):
        super(EvaluateZeroShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.class_model = self.kwargs.pop("class_model")

        self.metric_name = f'{self.prefix}0-shot_{self.k_way}-way_acc'

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
            x, y, attr = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(
                self.model,
                self.class_model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                attr,
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

        print(logs)


def fit(
        image_model: Module,
        class_model: Module,
        optimiser: Optimizer,
        loss_fn: Callable,
        epochs: int,
        dataloader: DataLoader,
        prepare_batch: Callable,
        metrics: List[Union[str, Callable]] = None,
        callbacks: List[Callback] = None,
        verbose: bool =True,
        fit_function: Callable = proto_net_zero_shot_episode,
        fit_function_kwargs: dict = {}
):
    """Function to abstract away training loop.

    TODO
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(image_model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y, attr = prepare_batch(batch)

            loss, y_pred = fit_function(image_model, class_model, optimiser, loss_fn, x, y, attr, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(image_model, y_pred, y, metrics, batch_logs)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
