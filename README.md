# Few-shot-learning
Implementation of various transfer learning and few-shot learning techniques on image data from
the [Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1) and its [smaller version](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

# Setup
### Requirements

Listed in `requirements.txt` (incomplete).


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

*Each product is identified by an ID like 42431. You will find a map to all the products in styles.csv. From here, you can fetch the image for this product from images/42431.jpg and the complete metadata from styles/42431.json.

To get started easily, we also have exposed some of the key product categories and it's display name in styles.csv*


### Tests (optional)

After adding the datasets run `pytest` in the root directory to run tests. Importantly, there are tests for the integration with this existing [github repository](https://github.com/oscarknagg/few-shot) which implements several few-shot-learning algorithms already. Integration mainly concerns handling of the fashion dataset. Make sure you to clone/fork the above github repository and that your python can find it.


### Transfer Learning

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


**Arguments**
- (TODO)

### Results

(TODO)


### Few-shot-learning

(TODO)
