import tensorflow as tf

#@tf.function
def rle2mask(rle, mask_shape):
    '''
    Converts a run lenght encoding (RLE) into a mask of shape mask_shape
    Args:
        rle: (str or bytestring) run lenght encoding. A series of space
            separated start-pixel run pairs.
        mask_shape: (tuple of 2 ints) the 2D expected shape of the mask
    Returns: mask of shape mask_shape
    '''
    size = tf.math.reduce_prod(mask_shape)

    s = tf.strings.split(rle)
    s = tf.strings.to_number(s, tf.int32)

    starts = s[0::2] - 1
    lens = s[1::2]

    total_ones = tf.reduce_sum(lens)
    ones = tf.ones([total_ones], tf.int32)

    r = tf.range(total_ones)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)

    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    mask = tf.reshape(mask_flat, (mask_shape[1], mask_shape[0]))
    return tf.transpose(mask)

#@tf.function(experimental_relax_shapes=True)
# def build_mask_array(rle, mask_size, n_classes=1):
#     '''
#     Converts a RLE or a list of RLEs, into an array of
#     shape [*mask_size, n_classes]
#     '''
#     if n_classes == 1:
#         mask = rle2mask(rle, mask_size)
#         mask = tf.expand_dims(mask, axis=2)
#     else:
#         ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
#         for i in tf.range(n_classes):
#             i = tf.cast(i, dtype=tf.int32)
#             ta = ta.write(i, tf.transpose(rle2mask(rle[i], mask_size)))
#         mask = tf.transpose(ta.stack())
#     mask = tf.reshape(mask, (*mask_size, n_classes))
#     return mask

#@tf.function
def build_mask_array(rle, mask_size, n_classes=1):
    '''
    Converts a RLE or a list of RLEs, into an array of
    shape (*mask_size, n_classes)
    '''
    if n_classes == 1:
        mask = rle2mask(rle, mask_size)
        mask = tf.expand_dims(mask, axis=2)
    else:
        mask = [rle2mask(rle[i], mask_size) for i in range(n_classes)]
        mask = tf.stack(mask, axis = -1)
    mask = tf.reshape(mask, (*mask_size, n_classes))
    return mask

#@tf.function
def mask2rle(mask):
    '''
    Converts a mask to a run lenght encoding (RLE) bytestring
    Args:
        mask: a numpy or tensorflow 2D mask array: 1 - mask, 0 - background
    Returns: RLE bytestring
    '''
    pixels = tf.transpose(mask)
    pixels = tf.reshape(pixels, [-1])
    pixels = tf.cast(pixels, dtype=tf.int64)
    pixels = tf.concat(([0], pixels, [0]), axis = 0)
    changes = (pixels[1:] != pixels[:-1])
    runs = tf.where(changes) + 1
    runs = tf.squeeze(runs)
    lens = runs[1::2] - runs[::2]

    zeros = tf.math.multiply(lens, 0)
    ones = tf.math.add(zeros, 1)
    inter = tf.stack((ones, zeros), axis = 1)
    inter = tf.reshape(inter, [-1])

    starts = tf.math.multiply(runs, inter)
    lens = tf.stack((zeros, lens), axis = 1)
    lens = tf.reshape(lens, [-1])

    rle = tf.math.add(starts, lens)
    rle = tf.strings.as_string(rle)
    rle = tf.strings.reduce_join(rle, separator=' ')
    return rle
