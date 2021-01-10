
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

