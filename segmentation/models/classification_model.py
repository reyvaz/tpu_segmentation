import tensorflow as tf
import tensorflow.keras.layers as L
from .base_cnns import get_cnn_func

def build_classifier(base_name, n_classes, input_shape = (None, None, 3),
                     weights ='imagenet', head_dropout=None, name_suffix=''):
    '''
    Assembles an CNN classifier
    Args:
        base_name: (str) one of 'efficientnetb0' to 'efficientnetb7' or
            'resnet50' to 'resnet152'.
        n_classes: (int) number of classes. i.e. number of filters in the output layer.
        weights: pretrained weights. one of None, 'imagenet',
            'noisy-student' or the path to the weights file to be loaded.
            Note: 'noisy-student' is only available for efficientnet classifiers.
        head_dropout: None or float. If float, the dropout rate before the last
            dense layer.
        name_suffix: (str) string to add to the model name
    '''
    CNN = get_cnn_func(base_name)
    base = CNN(input_shape=input_shape, weights=weights, include_top=False)

    x = L.GlobalAveragePooling2D()(base.output)
    if head_dropout:
        x = L.Dropout(head_dropout)(x)
    x = L.Dense(n_classes, activation='sigmoid', name='output_layer')(x)
    model = tf.keras.Model(inputs=base.input, outputs=x,
                           name='{}{}'.format(base.name.title(), name_suffix))
    return model
