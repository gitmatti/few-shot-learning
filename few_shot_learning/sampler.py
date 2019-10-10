import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data


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