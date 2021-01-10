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
    Calculates dice coefficients for each image and returns the average
    '''
    img_dice = tf.vectorized_map(lambda args: dice_coef(*args), [y_true, y_pred])
    img_dice_avg = tf.math.reduce_mean(img_dice)
    return img_dice_avg

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
