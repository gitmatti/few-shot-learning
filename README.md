# Few-shot-learning
Implementation of various transfer learning and few-shot learning techniques on image data from
the [Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1) and its [smaller version](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

# Setup
### Requirements

Listed in `requirements.txt`.


### Data

Edit the `DATA_PATH` variable in `config.py` to the location where
you the FashionProductImages dataset will be located.

Download the data from the links above (Kaggle sign-in necessary) and extract to `DATA_PATH/fashion-dataset` or `DATA_PATH/fashion-product-images-small` for the small dataset. There may be problems extracting the full dataset as the zipfile appears to be corrupeted.

The folder structure for the full data should look like this
```
DATA_PATH/
    fashion-dataset/
        images/
        styles/
        images.csv
        styles.csv
```

The `images/` folder contains 44441 JPG-images of 143 product classes. From the dataset description:

_Each product is identified by an ID like 42431. You will find a map to all the products in styles.csv. From here, you can fetch the image for this product from images/42431.jpg and the complete metadata from styles/42431.json. To get started easily, we also have exposed some of the key product categories and it's display name in styles.csv_


### Tests (optional)

After adding the datasets run `pytest` in the root directory to run tests. Importantly, there are tests for the integration with this existing [github repository](https://github.com/oscarknagg/few-shot) which implements several few-shot-learning algorithms already. Integration mainly concerns handling of the fashion dataset. Make sure you to clone/fork the above github repository and that your python can find it.


# Transfer Learning

Run `few_shot_learning/transfer.py` for running transfer learning on the fashion data with pre-trained ImageNet models.
The script will instantiate a pre-trained model and, in a first phase, fine-tune the weights for a 20-way classification task for the classes with the most samples in the dataset. These are given below. In a second fine-tuning phase, the weights are adapted by learning on a 123-way classification task for the remaining classes and samples.

| articleType           | count  |
|-----------------------|--------|
| Jeans                 | 609    |
| Perfume and Body Mist | 614    |
| Formal Shoes          | 637    |
| Socks                 | 686    |
| Backpacks             | 724    |
| Belts                 | 813    |
| Briefs                | 849    |
| Sandals               | 897    |
| Flip Flops            | 916    |
| Wallets               | 936    |
| Sunglasses            | 1073   |
| Heels                 | 1323   |
| Handbags              | 1759   |
| Tops                  | 1762   |
| Kurtas                | 1844   |
| Sports Shoes          | 2036   |
| Watches               | 2542   |
| Casual Shoes          | 2846   |
| Shirts                | 3217   |
| Tshirts               | 7070   |

Run e.g.

```
python -m experiments.transfer_experiment -a resnet50 -p 50 --distributed --date --epochs 100
```


**Arguments**

Inspect arguments via

```
python -m experiments.transfer_learning -h
```

Output:

```
usage: -m [-h] [--data DIR] [-a ARCHITECTURE] [-j N] [--epochs N] [-b N]
          [--lr LR] [--lr_tr LR] [--optim OPTIMIZER] [--optim-args DICT]
          [-p N] [--resume PATH] [-e] [--seed SEED] [--gpu GPU]
          [--distributed] [--device DEV] [--dtype DTYPE] [--date]
          [--small-dataset]

PyTorch Transfer Learning

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  -a ARCHITECTURE, --arch ARCHITECTURE
                        model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn | wide_resnet101_2 |
                        wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  -b N, --batch-size N  mini-batch size (default: 64)
  --lr LR, --learning-rate LR
                        initial learning rate
  --lr_tr LR, --learning-rate-tr LR
                        initial learning rate for transfer
  --optim OPTIMIZER     optimizer from torch.optim
  --optim-args DICT     optimizer args
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --distributed         Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs.
  --device DEV
  --dtype DTYPE
  --date                Create log and model folder with current date
  --small-dataset       Use dataset with smaller image size
```



### Results

Refer to `notebook/transfer_learning.ipynb` for a detailed description of results and strategy.

| Finetuning (Top-20) - Test Set Accuracy |       Top1    | Top5 |
|-------------------------------------------------------|------|-----|
| average across classes                                | 88.3 | 95.4|
| average across classes (w/o 'Perfume and Body Mist')  | 92.1 | 99.4|


| Transfer (Remaining classes) Test Set Accuracy |       Top1    | Top5 |
|-------------------------------------------------------|------|-----|
| average across classes                                | 46.8 | 60.5|
| average across classes (w/o test-only classes)        | ?    | ?   |


# Few-shot-learning

### Results

Refer to `notebook/few_shot_learning.ipynb` for a detailed description of results and strategy.

|                           | Fashion Small |     |      |      |     |      |
|---------------------------|---------------|-----|------|------|-----|------|
| **k-way**                 | **2**         |**5**|**15**|**2** |**5**|**15**|
| **n-shot**                | **1**         |**1**|**1** |**5** |**5**|**5** |
| 80 epochs                 | 89.9          |76.5 |58.5  |95.1  |88.7 |76.4  |
| best model (validation)   | 89.8          |73.0 |?     |95.6  |86.5 |?     |
