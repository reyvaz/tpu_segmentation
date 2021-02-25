import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    den = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    dice = (2. * intersection + smooth)/den
    return dice

def dice_avg(y_true, y_pred):
    '''
    Calculates the image-wise average dice coefficient in a batch of masks.
    args:
        y_true: tensor of shape (N, H, W, C), where N is the batch size and C
                is the number of classes (i.e. num of channels of the mask)
        y_pred: tensor of shape y_true.shape
    Returns the average dice coefficient accross images in the batch.
    '''
    img_dice = tf.vectorized_map(lambda args: dice_coef(*args), [y_true, y_pred])
    img_dice_avg = tf.math.reduce_mean(img_dice)
    return img_dice_avg

def channel_avg_dice(y_true, y_pred, smooth = 1.):
    '''
    Calculates the channel-wise average dice coefficient in a batch of masks.
    args:
        y_true: tensor of shape (N, H, W, C), where N is the batch size and C
                is the number of classes (i.e. num of channels of the mask)
        y_pred: tensor of shape y_true.shape
    Returns the average dice coefficient accross all channels.
    '''
    # Stacking the channels along the 1st dimension.
    # They will be ordered by channel_num then by image i.e. y_true[2] will be
    # the channel 0 of image 2
    y_true = tf.concat(tf.unstack(y_true, axis=-1), axis=0)
    y_pred = tf.concat(tf.unstack(y_pred, axis=-1), axis=0)

    channel_dice = tf.vectorized_map(lambda args: dice_coef(*args, smooth),
                                     [y_true, y_pred])
    channel_dice_avg = tf.math.reduce_mean(channel_dice)
    return channel_dice_avg

def dice_loss(y_true, y_pred, smooth = 1.):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = y_true_flat * y_pred_flat
    den = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    score = (2. * tf.reduce_sum(intersection) + smooth)/den
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def camvid_acc(y_true, y_pred):
    '''
    Requires all non-empty masks, as in camvid dataset
    '''
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    y_pred_flat = tf.round(y_pred_flat)
    mask_idxs = y_true_flat != 0
    mask_matches = (y_true_flat[mask_idxs] == y_pred_flat[mask_idxs])
    acc_camvid = tf.reduce_mean(tf.cast(mask_matches, dtype=tf.float32))
    return acc_camvid
