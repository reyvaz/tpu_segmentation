import tensorflow as tf
import tensorflow.keras.backend as K

# @tf.function
def random_zoom_out_and_pan(image, image_size, mask = None,
                            n_channels = 3, n_classes = 1,
                            zoom_range = (0.7, 0.9)):
    '''
    Applies random zoom out with random pan to an image.
    Args:
        image: a 3D image array
        image_size: a tuple (H, W). Height and width of both, the mask and image
        mask: a 3D mask array to apply the same transform as the image
        n_channels: (int) number of image channels typically 1 or 3
        n_classes: (int) one of {1, 3, 4} number of masks in the mask array. i.e
            the number of channels in the mask
        zoom_range: a tuple of floats (F1, F2) where F1<F2 and both are between
            0 and 1. The range of zoom-out factor.
    Returns:
        An image (and mask if specified) with random zoom-out + pan applied of
        image_size size.
    '''
    # Determine the random zoomed-out size
    random_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])
    height_width = tf.constant(image_size)
    dim_deltas = tf.cast(height_width, dtype=tf.float32)*(1-random_factor)
    dim_deltas = tf.cast(dim_deltas, dtype = tf.int32)
    zoomed_size = height_width - dim_deltas

    # Determine the random padding for each side to create random pan
    paddings_c = tf.constant([0,0])
    pad_top = tf.random.uniform([], 0, dim_deltas[0], dtype = tf.int32)
    pad_left = tf.random.uniform([], 0, dim_deltas[1], dtype = tf.int32)
    pad_bottom = dim_deltas[0] - pad_top
    pad_right = dim_deltas[1] - pad_left
    aug_paddings = tf.stack([[pad_top,pad_bottom], [pad_left, pad_right], paddings_c], axis=0)

    # apply zoom-out and panning to image, then reshape.
    zoomed_img = tf.image.resize(image, zoomed_size)
    zoomed_img = tf.pad(zoomed_img, aug_paddings)
    zoomed_img = tf.reshape(zoomed_img, (*image_size, n_channels))

    if mask != None:
        zoomed_mask = tf.image.resize(mask, zoomed_size)
        zoomed_mask = tf.pad(zoomed_mask, aug_paddings)
        zoomed_mask = tf.reshape(zoomed_mask, (*image_size, n_classes))
        return zoomed_img, zoomed_mask
    else: return zoomed_img

def left_right_flip(image, mask):
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)
    return image, mask

def up_down_flip(image, mask):
    image = tf.image.flip_up_down(image)
    mask = tf.image.flip_up_down(mask)
    return image, mask

def diag_flip(image, mask):
    image, mask = left_right_flip(image, mask)
    image, mask = up_down_flip(image, mask)
    return image, mask

# @tf.function
def random_zoom_in(image, image_size, n_channels, zoom_range = (0.7, 0.9)):
    '''
    Applies random zoom-in of a random area in the image.
    Args:
        image: an image array
        image_size: a tuple (H, W)
        zoom_range: a tuple of floats (F1, F2) where F1<F2 and both are between 0
            and 1. The range of the zoom-in factor. i.e. a zoom-in factor of 0.9
            will randomly zoom-in an area of H*0.9 x W*0.9 of the original image

    Returns:
        An image with random zoom-in of a random image area
    '''
    h, w = image_size
    aspect_ratio = tf.math.divide(w, h)
    aspect_ratio = tf.cast(aspect_ratio, tf.float32)

    # Determine the random height and width of the area to be zoomed-in.
    random_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])
    dy = h*random_factor
    dx = dy*aspect_ratio
    dy = tf.cast(dy, tf.int32)
    dx = tf.cast(dx, tf.int32)

    # Determine the random position of the random area within the original image
    max_x = w - dx
    max_y = h - dy
    x = tf.random.uniform([], 0, max_x, dtype = tf.int32)
    y = tf.random.uniform([], 0, max_y, dtype = tf.int32)

    # Crop + Resize -> Zoom-in
    cropped_img = tf.image.crop_to_bounding_box(image, y, x, dy, dx)
    zoomed_img = tf.image.resize(cropped_img, size = image_size)
    zoomed_img = tf.reshape(zoomed_img, (*image_size, n_channels))
    return zoomed_img

# @tf.function
def image_mask_zoom_in(image, mask, image_size, label = None,
                               n_channels = 3, n_classes = 1,
                               zoom_range = (0.1, 0.4)):
    '''
    Applies random zoom-in to a random area in the image and the mask
    Args:
        image: a 3D image array of size (H, W, C), C is one of {1, 3, 4}
        mask: a 3D mask array of size (H, W, C), C is one of {1, 3, 4}
        image_size: a tuple (H, W)
        label: one of {None, 0, 1}, provide if available to reduce computations,
            0 if the mask is empty, 1 otherwise.
        n_channels: (int) number of image channels typically 1 or 3
        n_classes: (int) one of {1, 3, 4} number of masks in the mask array.
            i.e the number of channels of the mask. For implementing n_classes > 4, see
            stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images
        zoom_range: a tuple of floats (F1, F2) where F1<F2 and both are between 0
            and 1. The range of the zoom-in factor. A larger zoom-in factor will
            yield a more intense zoom-in.
    Returns:
        an image and mask tuple with mask preserving zoom-in applied to both
    '''
    if label != 0 and tf.math.reduce_sum(tf.round(mask)) > 0:

        # transform to prevent left-top bias
        P = tf.random.uniform([], dtype='float32')
        if P > 0.75: image, mask = left_right_flip(image, mask)
        elif P > 0.50: image, mask = up_down_flip(image, mask)
        elif P > 0.25: image, mask = diag_flip(image, mask)

        h, w = image_size
        aspect_ratio = tf.cast(tf.math.divide(w, h), tf.float32)

        if n_classes > 1: mask_ = tf.math.reduce_sum(tf.round(mask), axis = 2)
        else: mask_ = tf.round(mask)

        # locate top-left and bottom-right points from the mask, then create new
        # coordinates that cuts the distance between the mask and the edges of the
        # image by the random_factor.
        locations = tf.transpose(tf.where(mask_))
        locations = tf.cast(locations, dtype=tf.float32)

        random_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])

        x1 = tf.cast(random_factor*tf.math.reduce_min(locations[1]), tf.int32)
        y1 = tf.cast(random_factor*tf.math.reduce_min(locations[0]), tf.int32)

        x2 = tf.math.reduce_max(locations[1])
        y2 = tf.math.reduce_max(locations[0])
        x2 = tf.cast(x2 + (w - x2)*(1-random_factor), tf.int32)
        y2 = tf.cast(y2 + (h - y2)*(1-random_factor) , tf.int32)

        # calculate runs (distances), required by tf.image.crop_to_bounding_box()
        dy = tf.cast(y2 - y1, tf.float32)
        dx = tf.cast(x2 - x1, tf.float32)

        # fix to original aspect ratio, make sure the crop is within the orig
        # image margins
        if tf.math.divide(dx, dy) > aspect_ratio: dx = dy*aspect_ratio
        else: dy = tf.math.divide(dx, aspect_ratio)

        dy = tf.cast(dy, tf.int32)
        dx = tf.cast(dx, tf.int32)

        # fail-safe
        dy = tf.math.maximum(dy, 1)
        dx = tf.math.maximum(dx, 1)
        dy = tf.math.minimum(dy, h-y1)
        dx = tf.math.minimum(dx, w-x1)

        # crop, resize
        image = tf.image.crop_to_bounding_box(image, y1, x1, dy, dx)
        image = tf.image.resize(image, size = image_size)

        mask = tf.image.crop_to_bounding_box(mask, y1, x1, dy, dx)
        mask = tf.image.resize(mask, size = image_size)

        # revert flip transforms
        if P > 0.75: image, mask = left_right_flip(image, mask)
        elif P > 0.50: image, mask = up_down_flip(image, mask)
        elif P > 0.25: image, mask = diag_flip(image, mask)

        mask = tf.reshape(mask, (*image_size, n_classes))

    else:
        image = random_zoom_in(image, image_size, n_channels, zoom_range = (0.7, 0.9))

    image = tf.reshape(image, (*image_size, n_channels))
    return image, mask


# @tf.function
def rotate_img(image, dim, n_channels, degrees):
    '''
    Rotates an image by `degrees` degrees
    Args:
        image: array of one squared image of size (dim, dim, n_channels)
        n_channels: (int)
        degrees: (int or float) the angle of the rotation in degrees
    Returns:
        An image rotated by an angle of `degrees`
    '''
    xdim = dim%2

    degrees = degrees * tf.ones([1], dtype='float32')
    rotation = 3.141593 * degrees / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    m = get_3x3_mat([c1,   s1,   zero,
                     -s1,  c1,   zero,
                     zero, zero, one])

    # list destination pixel indices
    x = tf.repeat(tf.range(dim//2, -dim//2,-1), dim)
    y = tf.tile(tf.range(-dim//2, dim//2), [dim])
    z = tf.ones([dim*dim], dtype='int32')
    idx = tf.stack( [x,y,z] )

    # rotate destination pixels onto origin pixels
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -dim//2+xdim+1, dim//2)

    # find origin pixel values
    idx3 = tf.stack([dim//2-idx2[0,], dim//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[dim, dim, n_channels])

# @tf.function
def random_rotate(image, dim, n_channels=3, mask = None,
                  n_classes = 1, max_degrees = 25.):
    '''
    Applies random rotation to an image (and mask if indicated)
    Args:
        image: array of one squared image of shape (dim, dim, n_channels)
        dim: (int) the lenght of one of the image sides
        n_channels: (int) the number of channels of the image
        mask : array of one squared image of shape (dim, dim, n_classes)
        n_classes: (int) the number of classes (i.e. channels) of the mask array
        max_degrees: (float) the maximum rotation to be applied in degrees
    '''
    degrees = max_degrees * tf.random.uniform([], -1, 1, dtype='float32')
    image = rotate_img(image, dim, n_channels, degrees)

    if mask != None:
        mask = rotate_img(mask, dim, n_classes, degrees)
        return image, mask
    else: return image

#@markdown shear

#@tf.function
def shear_img(image, dim, n_channels, shear_factor = 7.0):
    '''
    Shears an image by `shear_factor` degrees
    Args:
        image: array of one squared image of size (dim, dim, n_channels)
        n_channels: (int)
        shear_factor: (int or float) the shear factor in degrees
    Returns:
        An sheared image
    '''
    xdim = dim%2
    shr = shear_factor * tf.ones([1], dtype='float32')
    shear    = 3.141593 * shr / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # shear matrix
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    m = get_3x3_mat([one,  s2,   zero,
                     zero, c2,   zero,
                     zero, zero, one])

    # list destination pixel indices
    x   = tf.repeat(tf.range(dim//2, -dim//2,-1), dim)
    y   = tf.tile(tf.range(-dim//2, dim//2), [dim])
    z   = tf.ones([dim*dim], dtype='int32')
    idx = tf.stack( [x,y,z] )

    # rotate destination pixels onto origin pixels
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -dim//2+xdim+1, dim//2)

    # find origin pixel values
    idx3 = tf.stack([dim//2-idx2[0,], dim//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d,[dim, dim, n_channels])

#@tf.function
def random_shear(image, dim, n_channels=3, mask = None,
                  n_classes = 1, max_shear = 7.):
    '''
    Applies random shear to an image (and mask if indicated)
    Args:
        image: array of one squared image of shape (dim, dim, n_channels)
        dim: (int) the lenght of one of the image sides
        n_channels: (int) the number of channels of the image
        mask : array of one squared image of shape (dim, dim, n_classes)
        n_classes: (int) the number of classes (i.e. channels) of the mask array
        max_degrees: (float) the maximum shear to be applied in degrees
    '''
    shear = max_shear * tf.random.uniform([], dtype='float32')
    image = shear_img(image, dim, n_channels, shear)

    if mask != None:
        mask = shear_img(mask, dim, n_classes, shear)
        return image, mask
    else: return image

# @tf.function
def coarse_dropout(image, image_size, n_chan, count_range=(20, 100), m_size = 0.01):
    '''
    Applies coarse dropout to an image
    Args:
        image: an image array of shape (H, W, C)
        image_size: a tuple of ints (H, W)
        n_chan: (int) number of channels
        count_range: a tuple of ints. min and max for the random dropout counts
        m_size: the size of the dropout squares relative to image height (H)
    Returns:
        An image with coarse dropout applied
    '''
    _count = tf.random.uniform([], count_range[0], count_range[1], dtype=tf.int32)
    ydim, xdim = image_size
    width = tf.cast(m_size*ydim, tf.int32)

    # coarse dropout prob
    for k in range(_count):
        # choose random location
        x = tf.cast(tf.random.uniform([],0,xdim),tf.int32)
        y = tf.cast(tf.random.uniform([],0,ydim),tf.int32)

        # compute square
        ya = tf.math.maximum(0, y-width//2)
        yb = tf.math.minimum(ydim, y+width//2)
        xa = tf.math.maximum(0, x-width//2)
        xb = tf.math.minimum(xdim, x+width//2)

        # apply dropout
        one = image[ya:yb, 0:xa, :]
        two = tf.zeros([yb-ya, xb-xa, n_chan])
        three = image[ya:yb, xb:xdim, :]
        middle = tf.concat([one, two, three], axis=1)
        image = tf.concat([image[0:ya,:,:], middle, image[yb:ydim,:,:]], axis=0)
        image = tf.reshape(image, [ydim, xdim, n_chan])
    return image
