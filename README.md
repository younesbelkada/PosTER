# PosTER

Implementation of PosTER - Pose Transformer Encoder Representation - for pedestrian attributes recognition. Object Recognition and Computer Vision course. ENS Paris Saclay 2021-2022

![](images/architecture.png)

> We introduce in a transformer architecture pretrained in a self-supervised setting to generate meaningful and interpretable embeddings for poses. Those embeddings can then be used to make attribute prediction with a simple decoder reaching comparable results with models trained from scratch on some attributes.

## Requirements

This repository requires mainly ```pytorch``` and ```wandb``` for visualization. We strongly recommend you to create an account on [wandb](https://wandb.ai/) for better visualization. For installing the libraries, run:

```
pip3 install -r requirements.txt
```

## Get predictions

### Model zoo

+ PosTER pre-trained on TITAN and fine-tuned on TITAN attributes: [here](https://drive.google.com/file/d/1-BhkcTJ7wXNwh1HX5bUGPeZVlWyJ626P/view?fbclid=IwAR2fLV4xLEHYfj0ia2gnJBUE84QFv1e50S3RM9o5iscnLqM4JLjixSuOaB0)
+ Other models coming soon..!

Once the model downloaded, put your input images (```png```or ```jpg```) into the ```input/``` directory and the pre-trained model that has to be renamed ```PosTER_FT.p``` into ```Models```. Then, run ```python3 predict.py```. The output image should be save at ```output/```


## Tokenizing Poses

![](images/tokenization.drawio.png)

We tokenize poses by individual body part. Therefore, a single pose containing 17 keypoints can be seen as a sequence of 17 token of dimension 3 in order to be fed to a Transformers based architecture.

## Augmenting Poses

We introduce in ```Poster/Datasets/augmentations.py``` augmentations that can be applied to 2d Human poses, such as flipping, random translation and normalization.

## Pose modeling 

### Pose modeling on PIE dataset

We extracted poses using OpenPifPaf on PIE dataset. The 2D poses files can be found on this [link](https://drive.google.com/file/d/195g6eDeAaLRt7nEN5EweB7-eWwbktkQ_/view?usp=sharing). You should download it and extract it.

### Running pose-modeling script

>**You have one config file to change and manipulate**.

Please carefully refer to our [training wiki](https://github.com/younesbelkada/PosTER/wiki/Training-wiki) in order to understand each argument of the config file ```config.json```. 

After modifying the config file, run ```python test.py```.

#### Wandb visualization (Recommended)

We highly recommend you to use wandb for visualzing intermediate results while training your model. You can easily create an account on [wandb](wandb.ai) and push the results using your credentials. Read our [training wiki](https://github.com/younesbelkada/PosTER/wiki/Training-wiki) to understand more.