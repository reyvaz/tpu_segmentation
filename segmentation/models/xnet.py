import tensorflow as tf
import tensorflow.keras.layers as L
from .base_cnns import get_cnn_func
from .builder_utils import *

xnet_skip_connections_dict = {
    'efficientnet': ('block6a_expand_activation', 'block4a_expand_activation',
                     'block3a_expand_activation', 'block2a_expand_activation',
                     'top_activation', 'block5a_expand_activation',
                     'block3b_expand_activation', 'block2b_expand_activation'),

    'resnet':       ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out',
                     'conv1_relu','conv5_block3_out', 'conv4_block1_out',
                     'conv3_block1_out', 'conv2_block1_out')
    }

# Unet++ Upsample Blocks
def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = L.Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = L.BatchNormalization(name=bn_name)(x)
        x = L.Activation('relu', name=relu_name)(x)
        # x = L.LeakyReLU(alpha=0.1, name=relu_name+'_leaky')(x)
        return x
    return layer

def Upsample2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = layer_names(stage, cols)
        x = L.UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list: x = L.Concatenate(name=merge_name)([x] + skip)
            else: x = L.Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

def Transpose2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = layer_names(stage, cols)
        x = L.Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = L.BatchNormalization(name=bn_name+'1')(x)
        x = L.Activation('relu', name=relu_name+'1')(x)
        # x = L.LeakyReLU(alpha=0.1, name=relu_name+'1_leaky')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = L.Concatenate(name=merge_name)(merge_list)
            else:
                x = L.Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

# Unet++ Model
def xnet_model(backbone, num_classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='transpose',
               activation='sigmoid',
               use_batchnorm=True):
    '''
    Assembles a tf.keras.Model based Unet++ using a prebuild encoder as input.
    Args:
        backbone: a tf.keras.Model instance to use as the Unet++ encoder.
        num_classes: (int) number of classes. i.e. number of filters in the output layer.
        skip_connection_layers: a list of layer numbers or names.
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
        tf.keras.Model: UNet++ Zhou, Zongwei et al. (2019)
                        (https://arxiv.org/abs/1912.05074)

    '''

    input = backbone.input

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    if len(skip_connection_layers) > n_upsample_blocks:
        downsampling_layers = skip_connection_layers[int(len(skip_connection_layers)/2):]
        skip_connection_layers = skip_connection_layers[:int(len(skip_connection_layers)/2)]
    else:
        downsampling_layers = skip_connection_layers

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]
    downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in downsampling_layers])
    downsampling_list = [backbone.layers[downsampling_idx[i]].output for i in range(len(downsampling_idx))]
    downterm = [None] * (n_upsample_blocks+1)
    for i in range(len(downsampling_idx)):
        if downsampling_list[0].name == backbone.output.name:
            downterm[n_upsample_blocks-i] = downsampling_list[i]
        else:
            downterm[n_upsample_blocks-i-1] = downsampling_list[i]
    downterm[-1] = backbone.output

    interm = [None] * (n_upsample_blocks+1) * (n_upsample_blocks+1)
    for i in range(len(skip_connection_idx)):
        interm[-i*(n_upsample_blocks+1)+(n_upsample_blocks+1)*(n_upsample_blocks-1)] = skip_layers_list[i]
    interm[(n_upsample_blocks+1)*n_upsample_blocks] = backbone.output

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks-j):
            upsample_rate = to_tuple(upsample_rates[i])

            if i == 0 and j < n_upsample_blocks-1 and len(skip_connection_layers) < n_upsample_blocks:
                interm[(n_upsample_blocks+1)*i+j+1] = None
            elif j == 0:
                if downterm[i+1] is not None:
                    interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2],
                                      i+1, j+1, upsample_rate=upsample_rate,
                                      skip=interm[(n_upsample_blocks+1)*i+j],
                                      use_batchnorm=use_batchnorm)(downterm[i+1])
                else:
                    interm[(n_upsample_blocks+1)*i+j+1] = None
            else:
                interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-2],
                                  i+1, j+1, upsample_rate=upsample_rate,
                                  skip=interm[(n_upsample_blocks+1)*i : (n_upsample_blocks+1)*i+j+1],
                                  use_batchnorm=use_batchnorm)(interm[(n_upsample_blocks+1)*(i+1)+j])

    x = L.Conv2D(num_classes, (3,3), padding='same', name='final_conv')(interm[n_upsample_blocks])
    x = L.Activation(activation, name=activation)(x)

    model = tf.keras.Model(input, x, name='{}-UnetPP'.format(backbone.name.title()))

    return model

# Unet++ Builder (for matching and unmatching weights)
def xnet(backbone_name, num_classes, input_shape=(None, None, 3),
         skip_connection_layers='default',
         n_skip_layers = 8,
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
    Assembles a tf.keras.Model based Unet++ with specified backbone.
    Args:
        backbone_name: (str) one of 'efficientnetb0' to 'efficientnetb7' or
            'resnet50' to 'resnet152'.
        num_classes: (int) number of classes. i.e. number of filters in the output layer.
        input_shape: shape of input data/image (H, W, C). Generally, (None, None, C)
            will suffice, however H and W in the input data need be multiples of 32.
        skip_connections: if 'default', it will use default skip connections,
            else provide a list of layer numbers or names.
        n_skip_layers: (int: 8 or 4) when `skip_connections` = 'default', the
            number of default skip connection layers to use.
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
            'noisy-student' or the path to the weights file to be loaded.
            Note: 'noisy-student' is only available for efficientnet encoders.
        weights_by_name: (bool) select True when weights' layers do not perfectly
            match `efficientnet` or `tf.keras.applications` functionals' layers.
            When True, it will load the weights by layer names.

    Returns:
        tf.keras.Model: UNet++ Zhou, Zongwei et al. (2019)
                        (https://arxiv.org/abs/1912.05074)

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
        skip_connection_layers = xnet_skip_connections_dict[skips_dict_key][:n_skip_layers]

    model = build_xnet(backbone, num_classes, skip_connection_layers,
                       decoder_filters, upsample_rates, n_upsample_blocks,
                       block_type, activation, use_batchnorm)
    return model
