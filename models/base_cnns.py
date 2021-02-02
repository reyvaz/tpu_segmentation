import tensorflow as tf
import efficientnet.tfkeras as efn
import re

cnn_dict = {
    'efficientnetb0': (efn.EfficientNetB0),
    'efficientnetb1': (efn.EfficientNetB1),
    'efficientnetb2': (efn.EfficientNetB2),
    'efficientnetb3': (efn.EfficientNetB3),
    'efficientnetb4': (efn.EfficientNetB4),
    'efficientnetb5': (efn.EfficientNetB5),
    'efficientnetb6': (efn.EfficientNetB6),
    'efficientnetb7': (efn.EfficientNetB7),
    'efficientnetL2': (efn.EfficientNetL2),
    'resnet50':       (tf.keras.applications.ResNet50),
    'resnet101':      (tf.keras.applications.ResNet101),
    'resnet152':      (tf.keras.applications.ResNet152)
    }

def get_cnn_func(cnn_name):
    if isinstance(cnn_name, int):
        if cnn_name < 8:
            cnn_key = 'efficientnetb{}'.format(cnn_name)
        else: cnn_key = 'resnet{}'.format(cnn_name)

    else: cnn_key = re.sub('[\W_]', '', cnn_name).lower()
    return cnn_dict[cnn_key]
