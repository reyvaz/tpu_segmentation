
# TPU Segmentation Models and Utils (WORK IN PROGRESS Jan 8, 2021)


## Contains TPU compatible tf.keras based segmentation models and utils. 

#### Builders for:

- Unet [(Ronneberger, Olaf et al., 2015)](https://arxiv.org/abs/1505.04597)
- Unet++ [(Zhou, Zongwei et al., 2019)](https://arxiv.org/abs/1912.05074)

With EfficientNet (B0 to B7) and Resnet (50, 101, and 152) backbones. 

#### Augmentation functions for image + mask not readily available in tensorflow, including:

- Random Zoom-in (mask preserving)
- Random Zoom-out with random pan
- Random Rotate
- Random Shear
- Random Coarse Dropout

#### RLE Encoding/Decoding fully implemented using tensorflow

- RLE to Mask
- Mask to RLE

#### Metrics/Loss functions:
- DICE coefficient (batch and average)
- DICE loss
- BCE DICE loss
- Camvid Accuracy (inspired by fast.ai)

#### Bonus
- Image Classification Builder with EfficientNet (B0 to B7) and Resnet (50, 101, and 152) as base CNNs. 

#### Installation and Use (from Python)
```
!git clone -q https://github.com/reyvaz/tpu_segmentation.git
!pip install -r tpu_segmentation/requirements.txt
from tpu_segmentation import *
```

To build Unet ++ 
```
model = xnet('EfficientNetB0', num_classes = 1, weights=None)
```

To build Unet
```
model = unet('EfficientNetB0', num_classes = 1, weights=None)
```

Notes:

- Models and functions are compatible with Colab and Kaggle TPUs running on their default TF versions, 2.4 and 2.2 respectively as of Jan, 9 2021.
- They also work on GPU and CPU anywhere with TF 2.2 to 2.4
- All augmentation functions can be `@tf.function` decorated. 
- Augmentations that apply transformations to the mask are currently limited to masks for 1, 3 or 4 one-hot encoded classes. 


**Code credits**: 

- Pavel Yakubovskiy's ([@qubvel](https://github.com/qubvel)) [EfficientNet](https://github.com/qubvel/efficientnet) and [Segmentation Models](https://github.com/qubvel/segmentation_models). The EfficientNet backbones are built using qubvel's efficientnet library. Imagenet and noisy-student weights come from there as well. Skip connections dictionaries are based on qubvel's Segmentation Models library. 

- Zongwei Zhou's ([@MrGiovanni](https://github.com/MrGiovanni)) [UNetPlusPlus library](https://github.com/MrGiovanni/UNetPlusPlus). Unet and Unet++ model builders as well as the upsampling related blocks were borrowed/adapted from MrGiovanni's UNetPlusPlus keras library. 

- Chris Deotte's [Triple Stratified KFold with TFRecords](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/notebook). The rotation and shear augmentation functions were adapted from Chris' code in this notebook. The code for the coarse dropout was also adapted from his code [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/169721).



**References**:
in progress

<br>















