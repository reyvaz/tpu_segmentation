import tensorflow as tf
import tensorflow.keras.layers as L
from .base_cnns import get_cnn_func
from .builder_utils import *

unet_skip_connections_dict = {
    'efficientnet': ('block6a_expand_activation', 'block4a_expand_activation',
                     'block3a_expand_activation', 'block2a_expand_activation'),
    'resnet':       ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out',
                     'conv1_relu')
    }

# Unet Blocks
def ConvNRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = L.Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = L.BatchNormalization(name=bn_name)(x)
        x = L.Activation('relu', name=relu_name)(x)
        #x = L.LeakyReLU(alpha=0.1,  name=relu_name)(x)
        return x
    return layer

def DecoderUpsample2DBlock(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = block_names(stage)
        x = L.UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        if skip is not None:
            x = L.Concatenate()([x, skip])

        x = ConvNRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)
        x = ConvNRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

def DecoderTranspose2DBlock(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = block_names(stage)
        x = L.Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = L.BatchNormalization(name=bn_name+'1')(x)
        x = L.Activation('relu', name=relu_name+'1')(x)
        if skip is not None:
            x = L.Concatenate()([x, skip])
        x = ConvNRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

# Unet Model
def unet_model(backbone, num_classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='transpose',
               activation='sigmoid',
               use_batchnorm=True):

    '''
    Assembles a tf.keras.Model based Unet using a prebuild encoder as input.
    Args:
        backbone: a tf.keras.Model instance to use as the Unet encoder.
        num_classes: (int) number of classes. i.e. the number of filters in the output layer.
        skip_connection_layers: if 'default', will use default skip connections,
            else provide a list of layer numbers or names.
        decoder_filters: (int) number of convolution layer filters in decoder blocks.
        n_upsample_blocks: (int) number of upsampling blocks.
        block_type: (str) one of 'transpose' or 'upsampling'.
        activation: (str) one of tf.keras activations for the output layer. e.g.
            Use 'sigmoid' for binary-segmentation or for multiclass segmentation
                with independent overlapping/non-overlapping class masks.
            Use 'softmax' for multiclass segmentation with non overlapping
                class masks (classes + background).
        use_batchnorm: (bool) if True, add batchnorm layer between `Conv2D` and
            `Activation` layers.

    Returns:
        tf.keras.Model: UNet Ronneberger, Olaf et al. (2015) (https://arxiv.org/abs/1505.04597)

    '''

    input = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = DecoderTranspose2DBlock
    else:
        up_block = DecoderUpsample2DBlock

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])
        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = L.Conv2D(num_classes, (3,3), padding='same', name='final_conv')(x)
    x = L.Activation(activation, name=activation)(x)

    model = tf.keras.Model(input, x, name='{}-Unet'.format(backbone.name.title()))
    return model

# Unet Builder (for matching and unmatching weights)
def unet(backbone_name, num_classes, input_shape=(None, None, 3),
         skip_connection_layers='default',
         decoder_filters=(256,128,64,32,16),
         upsample_rates=(2,2,2,2,2),
         n_upsample_blocks=5,
         block_type='transpose',
         activation='sigmoid',
         use_batchnorm=True,
         freeze_backbone = False,
         weights='imagenet',
         weights_by_name = False):

    '''
    Assembles a tf.keras.Model based Unet with specified backbone.
    Args:
        backbone_name: (str) one of 'efficientnetb0' to 'efficientnetb7' or
            'resnet50' to 'resnet152'.
        num_classes: (int) number of classes. i.e. number of filters in the output layer.
        input_shape: shape of input data/image (H, W, C). Generally, (None, None, C)
            will suffice, however H and W in the input data need be multiples of 32.
        skip_connection_layers: if 'default', will use default skip connections,
            else provide a list of layer numbers or names.
        decoder_filters: (int) number of convolution layer filters in decoder blocks.
        n_upsample_blocks: (int) number of upsampling blocks.
        block_type: (str) one of 'transpose' or 'upsampling'.
        activation: (str) one of tf.keras activations for the output layer. e.g.
            Use 'sigmoid' for binary-segmentation or for multiclass segmentation
                with independent overlapping/non-overlapping class masks.
            Use 'softmax' for multiclass segmentation with non overlapping
                class masks (classes + background).
        use_batchnorm: (bool) if True, add batchnorm layer between `Conv2D` and
            `Activation` layers.
        freeze_backbone: (bool) If true, will freeze the backbone weights.
        weights: pretrained weights for the encoder. one of None, 'imagenet',
            'noisy-student', or the path to the weights file to be loaded.
            Note: 'noisy-student' weights are only available for efficientnet encoders.
        weights_by_name: (bool) select True when weights' layers do not perfectly
            match `efficientnet` or `tf.keras.applications` functionals' layers.
            When True, it will load the weights by layer names.

    Returns:
        tf.keras.Model: UNet Ronneberger, Olaf et al. (2015) (https://arxiv.org/abs/1505.04597)

    '''
    if not weights_by_name: cnn_weights = weights
    else: cnn_weights = None

    CNN = get_cnn_func(backbone_name)
    backbone = CNN(include_top=False, weights=cnn_weights, input_shape=input_shape)

    if weights_by_name:
        backbone.load_weights(weights, by_name=True)

    backbone.trainable = not freeze_backbone

    if skip_connection_layers == 'default':
        if 'efficientnet' in backbone.name.lower():
            skips_dict_key = 'efficientnet'
        elif 'resnet' in backbone.name.lower():
            skips_dict_key = 'resnet'
        skip_connection_layers = unet_skip_connections_dict[skips_dict_key]

    model = unet_model(backbone, num_classes, skip_connection_layers,
                       decoder_filters, upsample_rates, n_upsample_blocks,
                       block_type, activation, use_batchnorm)
    return model
