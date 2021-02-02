
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

- Zongwei Zhou's ([@MrGiovanni](https://github.com/MrGiovanni)) [UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus). 

- Pavel Yakubovskiy's ([@qubvel](https://github.com/qubvel)) [EfficientNet](https://github.com/qubvel/efficientnet) and [Segmentation Models](https://github.com/qubvel/segmentation_models). 





- Chris Deotte's [Triple Stratified KFold with TFRecords](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/notebook). 

This repository was built upon Zongwei Zhou's [UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) and Pavel Yakubovskiy's [Segmentation Models](https://github.com/qubvel/segmentation_models) repositories. Code from their repositories were combined/modified to build Tensorflow 2.x (tf.keras), TPU-compatible Unet and Unet++ networks with backbones from  @qubvel's [EfficientNet](https://github.com/qubvel/efficientnet) library as well as ResNet backbones from `tf.keras.applications`. 

Specifically, Unet and Unet++ model builders as well as the upsampling related blocks were adapted from MrGiovanni's UNetPlusPlus keras library. The EfficientNet backbones are built using qubvel's efficientnet library. Imagenet and noisy-student weights come from there as well. Skip connections dictionaries are based on qubvel's Segmentation Models. 

The rotation and shear augmentation functions were adapted from Chris Deotte's code in this [notebook](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/notebook). The code for coarse dropout was also adapted from Chris' code in  [this discussion](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/169721).

**References**:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, & Jian Sun. (2015). Deep Residual Learning for Image Recognition.

Mingxing Tan, & Quoc V. Le. (2020). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.

Olaf Ronneberger, Philipp Fischer, & Thomas Brox. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

Zhou, Z., Siddiquee, M., Tajbakhsh, N., & Liang, J. (2019). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation IEEE - Transactions on Medical Imaging.




<br>















