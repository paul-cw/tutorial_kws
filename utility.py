import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import fspath
from pathlib import Path
import librosa
import IPython.display as ipd
import random
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from scikitplot.metrics import plot_confusion_matrix, plot_roc
#import keras
from tensorflow import keras
from datetime import datetime

## smooting function
def smoothing(x, w):
    """
    smoothes probabilities of each keyword over a window
    parameters:
    x: array xij of frames i, each frame containing probabilities for jth keyword xij
    w: size of the window we want to average over

    returns:
    array of smoothed probabilities of same length as x
    """

    res = np.zeros(shape=x.shape)

    y = np.transpose(x)
    # for a given kw i, calc the average depending on wether the jth frame has w (enough) neighbours to the left
    for i, yi in enumerate(y):
        for j, _ in enumerate(yi):
            vec_to_avg = yi[max(0, j - w + 1):j + 1]
            # print(vec_to_avg)
            res[j, i] = np.average(vec_to_avg)

    return res

## utility for converting labels to floats
def one_at(x, n_output_neurons=12):
    target = [0.] * n_output_neurons
    target[x] = 1.
    return target

## audio utility functions
def load_all_wavs_in_dir(direc, sr):
    '''loads all the .wav data in dir, using the sample rate sr'''
    wave_files_paths = [fspath(path) for path in Path(direc).rglob('*.wav')]
    print('found # of files:', len(wave_files_paths))
    wavs = [librosa.load(file_path, sr=sr) for file_path in wave_files_paths]
    tmp = np.array(wavs, dtype='object')
    tmp = tmp[:, 0]
    return tmp

def add_background_noise(y, background_frequency, noise_data, Ab=None, snr=None, clip=False):

    assert (Ab or snr != None)
    if np.random.uniform(0, 1) < background_frequency:
        y_noise = noise_data[np.random.randint(0, len(noise_data) - 1)]

        # choose appropriately long portion and add noise
        start = np.random.randint(0, y_noise.shape[0] - y.shape[0])
        y_noise = y_noise[start:start + y.shape[0]]

        if snr != None:
            yp = add_scaled_noise(y_noise, y, snr)
        else:
            yp = y + y_noise * np.random.uniform(0, Ab)

        if clip:  # limit values to [-1,1]
            np.clip(yp, -1., 1., out=yp)

        return yp
    else:
        return y

def augment_audio(silence=False, mode='training',
                  sig=[], fs=16000, padd_audio_to_samples=None,
                  time_shift_by_max=0.0,
                  background_frequency=0., noise_data=[], Ab=0.1, snr=20,
                  l=40, s=20, n_mfccs=40):
    ''' takes a signal and turns it into potentially augmented audio features (mfccs)

    Args:
        silence: flag for silence files. If true we want to augment the signal everytime
        mode: in 'training' mode we want to add background noise and apply a time sift
        sig: The audio signal as a list of floats
        fs: resample to fs
        padd_audio_to_samples: add 0s to a signal until it is of size padd_audio_to_samples
        time_shift_by_max: the maximum time the signal gets shifted
        bckground_frequency: [0,1]. Likelihood of mixing in background noise
        noise_data: a list of signals to mix into the signal
        Ab: Amplitude with which the background noise is added
        l: window size [ms]
        s: stride [ms]
        n_mfccs: number of mfcc coefficients
    Returns:
        n_mfcc>0: return augmented mfcc array with size n_windows x n_mfccs
        else: return augmented audio signal
    '''

    # audio padding
    if padd_audio_to_samples != None:
        sig = pad_signal(sig, padd_audio_to_samples)

    # apply a time shift of maximal time_shift_by_max seconds
    if time_shift_by_max != 0.0 and mode == 'training':
        sig = shift_by(y=sig, ts=time_shift_by_max, fs=fs)

    # add background noise
    if silence:  # always add background and set signal to 0!
        sig = add_background_noise(sig * 0., background_frequency=1., Ab=Ab, noise_data=noise_data)
        np.clip(sig, -1., 1., out=sig)
    if mode == 'training':
        if background_frequency > 0.:
            sig = add_background_noise(sig, background_frequency, Ab=Ab, noise_data=noise_data)

    # calculate mfcc features directly with librosa
    if n_mfccs > 0:
        lprime = l * int(fs / 1000)
        hprime = lprime - s * int(fs / 1000)
        feat = librosa.feature.mfcc(sig, win_length=lprime, n_mfcc=n_mfccs, sr=fs, n_fft=lprime, hop_length=hprime,
                                    center=False, power=2).T
        return feat
    else:
        return sig

def pad_signal(sig, padd_audio_to_samples=True):
    '''add 0s to left and right of the signal or keep only central portion'''
    if padd_audio_to_samples > len(sig):
        ## add 0s to left and right of the signal
        N = padd_audio_to_samples - len(sig)
        sig = np.pad(sig, (N // 2, N // 2 + N % 2), 'constant', constant_values=(0., 0.))
    elif padd_audio_to_samples < len(sig):
        ## keep only the central portion of size padd_audio_to_samples
        N = len(sig) - padd_audio_to_samples
        sig = sig[N // 2:]
        sig = sig[0:padd_audio_to_samples]
        
    return sig

def shift_by(y, ts, fs):
    '''randomly time shifts a signal y with sample frequency fs by up to +- ts'''
    
    ## copy the signal
    y_shift = np.copy(y)
    
    ## pick random shift value from [-ts, ts]
    ts = np.abs(ts)
    ts = np.random.uniform(-1.*ts, 1. * ts, 1)[0]
    shift_in_samples = int(np.abs(ts * fs))
    
    ## perform te shift (different for ts>0, ts<0)
    if ts > 0:
        ## add 0s to the left and take original signal length starting from sample 0:
        y_shift = np.pad(y_shift, (shift_in_samples, 0), mode='constant')[0:y_shift.shape[0]]
    elif ts < 0:
        ## add 0s to the right and take original signal length starting from shift in samples:
        y_shift = np.pad(y_shift, (0, shift_in_samples), mode='constant')[
                  shift_in_samples:y_shift.shape[0] + shift_in_samples]

    return y_shift

## utility for plotting results
class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir, roc=True):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir
        self.roc=roc

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = np.argmax(self.validation_data[1], axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true=y_true, y_pred=y_pred_class, ax=ax)  
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

        if self.roc:
            # plot and save roc curve
            fig, ax = plt.subplots(figsize=(16, 12))
            plot_roc(y_true, y_pred, ax=ax)
            fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
            plt.close('all')
        
## a residual block with flexible stride and number of filters
class ResBlock(layers.Layer):
    def __init__(self, n, s):
        super(ResBlock, self).__init__()

        if s == 2:
            self.in_layers = [
                layers.Conv2D(filters=n, kernel_size=[1, 1], strides=s, activation=None, use_bias=False,
                              padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
        elif s == 1:
            self.in_layers = []
        else:
            assert False

        self.mylayers = [
            layers.Conv2D(filters=n, kernel_size=[9, 1], strides=s, activation=None, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters=n, kernel_size=[9, 1], strides=1, activation=None, use_bias=False, padding='same'),
            layers.BatchNormalization()
        ]
        self.mylayers2 = layers.Add()
        self.mylayers3 = layers.ReLU()

    def get_config(self):
        config = super().get_config().copy()
        config.update({})
        return config

    def call(self, net):

        layer_in = net

        for lay in self.in_layers:
            layer_in = lay(layer_in)

        for lay in self.mylayers:
            net = lay(net)

        net = self.mylayers2([net, layer_in])
        net = self.mylayers3(net)
        return net
    
def keep_only_n_unknowns(df, unknown_percentage=100):
    """This function returns the original dataframe with the class weight of the unknown label reduced to unknown_percentage."""        
    dfu = df[df.keyword =='unknown']
    df  = df[df.keyword != 'unknown']

    ## keep only some unknowns
    N    = int(len(df) * unknown_percentage/100.)
    dfu = dfu.sample(n = N)
    
    df = pd.concat([df,dfu])
    df = df.sample(frac=1, random_state=0)
    
    print('new dataframe has ', N, 'unknown utterances')
    
    return df

def get_callbacks(output_dir = './', val_data=None, model=None, patience=25, min_delta=1e-10, roc=True):
    ## logging stuff
    if not os.path.isdir(output_dir + 'output'):
        os.mkdir(output_dir + 'output')
    output_path = output_dir + 'output/' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '/'
    os.mkdir(output_path)
    print('logging to: ', output_path)
    
    ## define callbacks for training
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=output_path + 'saved_models',
                                        monitor='val_categorical_accuracy',
                                        mode='max',
                                        save_best_only=True,
                                        run_eagerly=False),
        keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=patience, min_delta=min_delta,
                                      mode='max'),
        PerformanceVisualizationCallback(model, val_data, output_path + 'logs/imgs', roc=roc),
    ]
    
    return callbacks