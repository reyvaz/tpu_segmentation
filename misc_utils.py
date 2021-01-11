import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys, re

def model_parameter_count(model, return_counts = False):
    '''
    Breaks down and prints out the counts of parameters of a tf.keras model
    '''
    trainable_count = np.int(np.sum([K.count_params(w) for w in model.trainable_weights]))
    non_trainable_count = np.int(np.sum([K.count_params(w) for w in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count

    print('Total params: {:,}'.format(total_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    if return_counts: return total_count, trainable_count, non_trainable_count

def count_data_items(filenames):
    '''
    counts the number of data items when counts are written in the name of
    the .tfrec files
    '''
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def inspect_tfrecord(tfrec_fname, save_to_json=False):
    '''
    Prints out the contents of 1 TFRec example in a .tfrec file
    tfrec_fname: (str) a path to a .tfrec file
    save_to_json: (False or str), if str, it will be the prefix of the json file
    '''
    raw_dataset = tf.data.TFRecordDataset(tfrec_fname)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

    if save_to_json:
        original_stdout = sys.stdout
        with open('{}.json'.format(save_to_json), 'w') as f:
            sys.stdout = f
            print(example)
            sys.stdout = original_stdout
    return None
