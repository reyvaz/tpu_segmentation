import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from matplotlib import pyplot as plt

class metrics_improvements(tf.keras.callbacks.Callback):
    '''
    Prints out information when the value of the `watch` metric has improved.
    freq: (None or int), if int, the frequency (in epochs) to print out all
        metrics regardless of improvements.
    '''
    def __init__(self, watch='val_loss', mode = 'min', freq=None):
        self.freq = freq
        self.watch = watch
        self.mode = mode

    def on_train_begin(self, logs=None):
        if self.mode == 'min':
            self.best = np.Inf
            self.compare = tf.math.less
        else:
            self.best = -np.Inf
            self.compare = tf.math.greater
        self.best_logs = None
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.watch)
        if self.compare(current, self.best):
            print('New best at Epoch {:03d} {} improved from {:.4f} to {:.4f}'.format(
                epoch+1, self.watch, self.best, current))
            self.best = current
            self.best_logs = logs
            self.best_epoch = epoch

        if self.freq:
            if (epoch+1) % self.freq == 0:
                items = ['{}: {:.4f}'.format(i[0], i[1]) for i in logs.items()]
                print('Epoch: {:03d}'.format(epoch+1), *items)

    def on_train_end(self, logs=None):
        items = ['{}: {:.6f}'.format(i[0], i[1]) for i in self.best_logs.items()]
        print('\nBest at Epoch: {:03d}'.format(self.best_epoch+1), *items)

lr_default_params = {
    'lr_start': 3e-4,
    'lr_max': 3e-4,
    'lr_min': 1e-6,
    'lr_num_ramp_eps': 3,
    'lr_sustain_eps': 4,
    'lr_decay': 0.8}

lr_default_params =  list(lr_default_params.values())

def lrfn(epoch, lr_params = lr_default_params):
    lr_start, lr_max, lr_min, lr_ramp_ep, lr_sus_ep, lr_decay = lr_params
    if epoch < lr_ramp_ep: lr = (lr_max - lr_start)/lr_ramp_ep*epoch + lr_start
    elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
    else: lr = (lr_max - lr_min)*lr_decay**(epoch-lr_ramp_ep - lr_sus_ep)+lr_min
    return lr

def lr_schedule_builder(params=lr_default_params):
    lrsched = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lrfn(epoch, params), verbose=False)
    return lrsched

def plot_lr_timeline(lr_params, num_epochs = 25, show_list=False):
    '''
    Plots the learning rate timeline of for the lrfn() learning rate function
    '''
    lr_timeline = [lrfn(i, lr_params) for i in range(num_epochs)]
    t = [i+1 for i in range(num_epochs)]
    plt.figure(figsize=(max(10, int(num_epochs/3)), 4.5))
    plt.plot(t, lr_timeline)
    plt.xticks(t)
    plt.xlim(t[0], t[-1])
    idx = np.argmax(lr_params[:3])
    plt.ylim(0, lr_params[idx]*1.1)
    plt.title('Learning Rate Timeline')
    plt.show()
    if show_list: print(lr_timeline)

# lr_show prints out the current learning rate at the start of each epoch.
# it should be placed after LR-scheduler in the list of callbacks @ model.fit
lr_show = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin= lambda epoch,logs: print('lr: %e' % (K.eval(model.optimizer.lr))))

def config_checkpoint(filepath = 'weights.h5', monitor ='val_loss', mode = 'min', verbose=0):
    # configures the training checkpoint to save best weights
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = filepath,
        monitor = monitor,
        mode = mode,
        save_best_only = True,
        save_weights_only=True,
        verbose = verbose)
    return checkpoint
