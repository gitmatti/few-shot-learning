import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np

from few_shot.callbacks import Callback, CallbackList, DefaultCallback,\
    ProgressBarLogger, CSVLogger, ModelCheckpoint, LearningRateScheduler

from few_shot_learning.core_zero_shot import ZeroShotTaskSampler,\
    EvaluateZeroShot, prepare_zero_shot_task, proto_net_zero_shot_episode, \
    fit, setup_dirs
from few_shot_learning.datasets import FashionProductImages,\
    FashionProductImagesSmall
from few_shot_learning.models import ClassEmbedding
from few_shot_learning.utils_data import prepare_class_embedding,\
    prepare_vocab, prepare_vocab_embedding, prepare_word_embedding
from config import DATA_PATH, PATH

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def zero_shot_training(
        datadir=DATA_PATH,
        dataset='fashion',
        drop_lr_every=20,
        validation_episodes=200,
        evaluation_episodes=1000,
        episodes_per_epoch=100,
        n_epochs=80,
        small_dataset=True,
        k_train=5,
        k_test=5,
        q_train=5,
        q_test=1,
        embedding_features=512,
        distance='l2',
        pretrained=True,
        freeze=False,
        weight_decay=1e-4,
        monitor_validation=False,
        n_val_classes=10,
        architecture='resnet18',
):
    setup_dirs()

    if dataset == 'fashion':
        dataset_class = FashionProductImagesSmall if small_dataset \
            else FashionProductImages
    else:
        raise (ValueError, 'Unsupported dataset')

    param_str = f'{dataset}_nt=0_kt={k_train}_qt={q_train}_' \
                f'nv=0_kv={k_test}_qv={q_test}_small={small_dataset}_' \
                f'pretrained={pretrained}_validate={monitor_validation}'

    print(param_str)

    ###################
    # Create datasets #
    ###################

    # ADAPTED: data transforms including augmentation
    resize = (80, 60) if small_dataset else (400, 300)

    background_transform = transforms.Compose([
        transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    if monitor_validation:
        if not n_val_classes >= k_test:
            n_val_classes = k_test
            print("Warning: `n_val_classes` < `k_test`. Take a larger number"
                  "of validation classes next time. Increased to `k_test`"
                  "classes")

        # class structure for background (training), validation (validation),
        # evaluation (test): take a random subset of background classes
        validation_classes = list(
            np.random.choice(dataset_class.background_classes, n_val_classes))
        background_classes = list(
            set(dataset_class.background_classes).difference(
                set(validation_classes)))

        # use keyword for evaluation classes
        evaluation_classes = 'evaluation'

        # Meta-validation set
        validation = dataset_class(datadir, split='all',
                                   classes=validation_classes,
                                   transform=evaluation_transform,
                                   return_class_attributes=True)
        validation_sampler = ZeroShotTaskSampler(validation,
                                                 validation_episodes,
                                                 k_test, q_test)
        validation_taskloader = DataLoader(
            validation,
            batch_sampler=validation_sampler,
            num_workers=8
        )
    else:
        # use keyword for both background and evaluation classes
        background_classes = 'background'
        evaluation_classes = 'evaluation'

    # Meta-training set
    background = dataset_class(datadir, split='all',
                               classes=background_classes,
                               transform=background_transform,
                               return_class_attributes=True)
    background_sampler = ZeroShotTaskSampler(background, episodes_per_epoch,
                                             k_train, q_train)
    background_taskloader = DataLoader(
        background,
        batch_sampler=background_sampler,
        num_workers=8
    )

    # Meta-test set
    evaluation = dataset_class(datadir, split='all',
                               classes=evaluation_classes,
                               transform=evaluation_transform,
                               return_class_attributes=True)
    # ADAPTED: in the original code, `episodes_per_epoch` was provided to
    # `NShotTaskSampler` instead of `evaluation_episodes`.
    evaluation_sampler = ZeroShotTaskSampler(evaluation, evaluation_episodes,
                                             k_test, q_test)
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=evaluation_sampler,
        num_workers=8
    )

    ###############
    # Image Model #
    ###############
    model = models.__dict__[architecture](pretrained=pretrained)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, embedding_features)
    model.to(device)

    ###############
    # Class Model #
    ###############
    target_vocab = prepare_vocab(background.df_meta, columns=["articleType"])
    vocab_embedding, word2idx = prepare_vocab_embedding(target_vocab)
    # TODO figure out which classes
    train_embedding = prepare_class_embedding(background.classes,
                                              vocab_embedding, word2idx)
    eval_embedding = prepare_class_embedding(evaluation.classes,
                                             vocab_embedding, word2idx)

    # class_model = get_class_encoder(in_features, embedding_features)
    embed_dim = 300
    attribute_dim = background.attribute_features
    class_model = ClassEmbedding(embedding_features, train_embedding,
                                 eval_embedding, embed_dim, attribute_dim)

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
    optimiser = Adam(
        [param for param in model.parameters() if param.requires_grad]
        + [param for param in class_model.parameters()],
        lr=1e-3, weight_decay=weight_decay)
    loss_fn = torch.nn.NLLLoss().to(device)

    callbacks = [
        # ADAPTED: this is the test monitoring now - and is only done at the
        # end of training.
        EvaluateZeroShot(
            eval_fn=proto_net_zero_shot_episode,
            num_tasks=evaluation_episodes,  # THIS IS NOT USED
            k_way=k_test,
            q_queries=q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_zero_shot_task(k_test, q_test),
            distance=distance,
            on_epoch_end=True,  # TODO
            on_train_end=False,  # TODO
            prefix='val_',  # TODO
            class_model=class_model
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_0-shot_{k_test}-way_acc',
            verbose=1,  # ADAPTED
            save_best_only=False  # ADAPTED
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
    ]

    if monitor_validation:
        callbacks.append(
            # ADAPTED: this is the validation monitoring now - computed
            # after every epoch.
            EvaluateZeroShot(
                eval_fn=proto_net_zero_shot_episode,
                num_tasks=evaluation_episodes,  # THIS IS NOT USED
                k_way=k_test,
                q_queries=q_test,
                # BEFORE taskloader=evaluation_taskloader,
                taskloader=validation_taskloader,  # ADAPTED
                prepare_batch=prepare_zero_shot_task( k_test, q_test),
                distance=distance,
                on_epoch_end=True,  # ADAPTED
                on_train_end=False,  # ADAPTED
                prefix='val_',
                class_model=class_model
            )
        )

    fit(
        model,
        class_model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_zero_shot_task(k_train, q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_zero_shot_episode,
        fit_function_kwargs={'k_way': k_train, 'q_queries': q_train,
                             'train': True, 'distance': distance},
    )