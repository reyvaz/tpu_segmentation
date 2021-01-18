import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from .augmentations import *

axes_color = '#999999'
mpl.rcParams.update({'text.color' : "#999999", 'axes.labelcolor' : axes_color,
                     'font.size': 10, 'xtick.color':axes_color,'ytick.color':axes_color,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': axes_color, 'axes.linewidth':1.0, 'figure.figsize':[8, 4]})

def retrieve_examples(dataset, n_batches = 5):
    '''
    Retrives example tuples from a batched dataset
    Args: dataset: a batched dataset i.e. BatchDataset
    Returns: a list of examples of lentght n_batches*batch_size
    '''
    examples = []
    for item in dataset.take(n_batches):
        examples += [[item[k][j] for k in range(len(item))] for j in range(len(item[0]))]
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

def plot_image_mask(img_mask_tuple, height = 5, cmap = 'bone', mask_alpha = 0.4):
    aspect_ratio = img_mask_tuple[0].shape[1]/img_mask_tuple[0].shape[0]
    image = tf.keras.preprocessing.image.array_to_img(img_mask_tuple[0])
    mask = tf.keras.preprocessing.image.array_to_img(img_mask_tuple[1])

    width = height*aspect_ratio*3
    plt.figure(figsize=(width, height))

    plt.subplot(1,3,1)
    plt.imshow(image, cmap=cmap)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask, alpha=mask_alpha, cmap=cmap)
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(image, cmap=cmap)
    plt.imshow(mask, alpha=mask_alpha, cmap=cmap)
    plt.title('Image + Mask')
    plt.axis('off')
    plt.show()
    return None

def show_augmentations(image, mask):
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
    masks = [pair[1] for pair in pairs]

    fig, axs = plt.subplots(1, len(pairs), figsize=(25, 4))
    for c in range(len(pairs)):
        ax = axs[c]
        ax.imshow(tf.keras.preprocessing.image.array_to_img(imgs[c]), cmap=plt.cm.bone)
        ax.imshow(tf.keras.preprocessing.image.array_to_img(masks[c]), alpha = 0.4, cmap = plt.cm.bone)
        ax.axis('off')
        ax.set_title(titles[c], fontdict={'fontsize': 13})
    return None
