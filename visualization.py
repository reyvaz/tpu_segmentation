import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from .augmentations import *

axes_color = '#999999'
mpl.rcParams.update({'text.color' : "#999999", 'axes.labelcolor' : axes_color,
                     'font.size': 10, 'xtick.color':axes_color,'ytick.color':axes_color,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': axes_color, 'axes.linewidth':1.0, 'figure.figsize':[8, 4]})

def retrieve_examples(dataset, num, idx = None, unbatch = True):
    '''
    idx: (int or list of ints) the index of the features to be extracted from
        the examples.
    '''
    if unbatch: dataset = dataset.unbatch()
    examples = []
    for item in dataset.take(num):
        examples.append(item)
    if idx != None:
        if isinstance(idx, int): examples = [row[idx] for row in examples]
        else:
            examples = [[row[i] for i in range(len(row)) if i in idx] for row in examples]
    return examples

def plot_array(array, height = 5, alpha = 1, cmap = 'bone', show = True):
    aspect_ratio = array.shape[1]/array.shape[0]
    width = height*aspect_ratio
    plt.figure(figsize=(width, height))
    try: plt.imshow(array, cmap=cmap, alpha=alpha)
    except:
        array = tf.keras.preprocessing.image.array_to_img(array)
        plt.imshow(array, cmap=cmap, alpha=alpha)
    plt.axis('off')
    if True: plt.show()
    return None

def contoured_mask(mask, rgb_color = (0, 0, 255), alpha = 0.2):
    rgb_color = np.array(rgb_color)/255
    epsilon = 10e-6
    mask = np.squeeze(mask+epsilon).round().astype(int)
    boundary = find_boundaries(mask).astype(float)
    mask_ = np.zeros((*mask.shape, 4))
    for i, c in enumerate(rgb_color):
        mask_[..., i] = c
    mask_[..., 3] = np.maximum(alpha*mask, boundary)
    return mask_

def plot_image_mask(img_mask_tuple, height = 2, cmap = 'bone',
                    mask_rgb = (204, 0, 153), mask_alpha = 0.4):
    '''
    mask_rgb: tuple or list of tuples. tuple(s) contain the rgb values for the
        mask(s). Supply a list of rgb tuples for color coded masks.
    cmap: one from plt.cm, it has an effect only when the image has 1 channel.
    '''
    H, W = img_mask_tuple[0].shape[:2]
    aspect_ratio = W/H
    image = tf.keras.preprocessing.image.array_to_img(img_mask_tuple[0])
    mask = img_mask_tuple[1]
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask_chans = mask.shape[-1]
        if isinstance(mask_rgb, tuple): mask_rgb = [mask_rgb]*mask_chans
        masks = [(contoured_mask(mask[..., c], mask_rgb[c], alpha = mask_alpha
                                 )) for c in range(mask_chans)]
    else:
        masks = [contoured_mask(mask, mask_rgb, alpha = mask_alpha)]

    width = height*aspect_ratio*2
    width = min(width, 15)
    fig = plt.figure(figsize=(width, height))

    plt.subplot(1,2,1)
    plt.imshow(image, cmap=cmap)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(image, cmap=cmap)
    for m in masks:
        plt.imshow(m)
    plt.title('Image + Mask')
    plt.axis('off')
    fig.tight_layout()
    plt.show()
    return None

# def plot_image_mask(img_mask_tuple, height = 5, cmap = 'bone',
#                     mask_rgb = (204, 0, 153), mask_alpha = 0.4):
#     aspect_ratio = img_mask_tuple[0].shape[1]/img_mask_tuple[0].shape[0]
#     image = tf.keras.preprocessing.image.array_to_img(img_mask_tuple[0])
#     mask = contoured_mask(img_mask_tuple[1], mask_rgb, alpha = mask_alpha)
#
#     width = height*aspect_ratio*2
#     fig = plt.figure(figsize=(width, height))
#
#     plt.subplot(1,2,1)
#     plt.imshow(image, cmap=cmap)
#     plt.title('Image')
#     plt.axis('off')
#
#     plt.subplot(1,2,2)
#     plt.imshow(image, cmap=cmap)
#     plt.imshow(mask)
#     plt.title('Image + Mask')
#     plt.axis('off')
#     plt.show()
#     return None


def show_augmentations(image, mask, mask_rgb = (204, 0, 153), mask_alpha = 0.4):
    '''
    Prob Remove. Too specific. Also, although it gets the augmentations when
    mask channels > 1, it does not plot them. Pointless to generalize.
    '''
    h, w, n_channels = image.shape
    size = (h, w)
    n_classes = np.array(mask).shape[2]
    label = None

    aug_hor_flip = left_right_flip(image, mask)
    aug_zoomed_in = image_mask_zoom_in(image, mask, size, label, n_channels, n_classes)
    aug_zoomed_out = random_zoom_out_and_pan(image, size, mask, n_channels, n_classes)

    aug_rotated = random_rotate(image, size, n_channels, mask, n_classes, 7.)
    aug_sheared = random_shear(image, size, n_channels, mask, n_classes, 7.)
    aug_coarsed = [coarse_dropout(image, size, n_channels, (100, 200), 0.015), mask]

    pairs = [[image, mask], aug_hor_flip, aug_zoomed_in, aug_zoomed_out, aug_rotated, aug_sheared, aug_coarsed]
    titles = ['Original', 'Horizontal Flip', 'Zoom-In', 'Zoom-Out',
              'Random Rotate', 'Random Shear', 'Coarse Dropout' ]
    imgs = [pair[0] for pair in pairs]
    masks = [contoured_mask(pair[1], mask_rgb, mask_alpha) for pair in pairs]

    fig, axs = plt.subplots(1, len(pairs), figsize=(25, 4))
    for c in range(len(pairs)):
        ax = axs[c]
        ax.imshow(tf.keras.preprocessing.image.array_to_img(imgs[c]), cmap=plt.cm.bone)
        ax.imshow(masks[c])
        ax.axis('off')
        ax.set_title(titles[c], fontdict={'fontsize': 13})
    return None
