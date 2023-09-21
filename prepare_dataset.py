## training hyperparameters

from config import *

import numpy as np
from datetime import datetime
from tensorflow import keras
from tqdm.auto import tqdm  
import librosa

## activate tqdm for pandas
tqdm.pandas()

## fix random seeds for tensorflow
tf.random.set_seed(0)

## load the raw audio data into memory
signals_train = df_train.file_path.progress_apply(lambda x: pad_signal(librosa.load(data_dir + x, sr=fs)[0],
                                                                    fs)).values
signals_val   = df_val.file_path.progress_apply(  lambda x: pad_signal(librosa.load(data_dir + x, sr=fs)[0],
                                                                    fs)).values
signals_test  = df_test.file_path.progress_apply( lambda x: pad_signal(librosa.load(data_dir + x, sr=fs)[0],
                                                                    fs)).values

## loading the labels
keywords_test  = df_test.label_one_hot.apply(lambda x: np.asarray(x).astype('float32')).values
keywords_val   = df_val.label_one_hot.apply(lambda x: np.asarray(x).astype('float32')).values
keywords_train = df_train.label_one_hot.apply(lambda x: np.asarray(x).astype('float32')).values

## we need to treat silence utterances differently, so we need to pass the silence label to the loader
silence_label = df_all[df_all.keyword == 'silence'].label_one_hot.iloc[0]#.unique()

## create a loader that calculates mfccs and provides batches of data, especially important later
class GSCLoader(tf.keras.utils.Sequence):
    ''' Loader provides batches of size batchsize with features x' and labels y where x' = f(x) '''
    
    def __init__(self, batchsize, x, y, f=None, silence_label=None):
        
        self.x = np.stack(x)
        self.y = np.stack(y)
        self.batchsize = batchsize
        self.indices   = np.arange(self.x.shape[0])
        self.f         = f
        self.silence_label = np.argmax(silence_label)
    
    ## return the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.x) / self.batchsize))

    ## return a batch of features, labels
    def __getitem__(self, idx):
        
        inds = self.indices[idx * self.batchsize:(idx + 1) * self.batchsize]
        features = np.array([self.f(silence=np.argmax(self.y[i]==self.silence_label), sig=self.x[i]) for i in inds])
        labels = np.array(self.y[inds])
        
        return features , labels

    ## shuffle the training data when done with one epoch
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        print('shuffling indices') 
        
## Define the function to calculate mfccs from the audio signal
def f(silence, sig):
    return augment_audio(silence, mode = '', sig=sig, fs=fs, l=l, s=s, n_mfccs=n_mfccs, padd_audio_to_samples=fs)
    
## Create the loaders with a batchsize that returns the whole dataset when the loader is called
train_loader = GSCLoader(f = f, batchsize = len(keywords_train), y = keywords_train, x = signals_train, 
                         silence_label=silence_label)
val_loader   = GSCLoader(f = f        , batchsize = len(keywords_val) , y = keywords_val,   x = signals_val, 
                         silence_label=silence_label)
test_loader  = GSCLoader(f = f        , batchsize = len(keywords_test), y = keywords_test,  x = signals_test, 
                         silence_label=silence_label)


def f_augment(silence, sig):
    mode = 'training'
    return augment_audio(silence, mode, sig, fs=fs, 
                              time_shift_by_max=time_shift_by_max,
                              background_frequency=background_frequency,
                              noise_data=noise_data,
                              Ab=Ab,
                              l=l, s=s, n_mfccs=n_mfccs, 
                              padd_audio_to_samples=fs)


## Set augmentation criteria
time_shift_by_max = 0.1  # randomized time shift [s]
background_frequency = 0.8  # how often is background folded in? 1 = always, 0 = never
Ab= 0.1  # background amplitude
Nfold = 3

## For the training set signals will be augmented aka mode=='training'
from utility import load_all_wavs_in_dir
noise_data = load_all_wavs_in_dir(direc=brn_directory, sr=fs)

## create a train loader that again returns the whole dataset in one batch, but applies f_augment this time
train_loader_augmented = GSCLoader(f = f_augment, batchsize = len(keywords_train), y = keywords_train, x = signals_train, 
                         silence_label=silence_label)

## Validation set
val_data = val_loader.__getitem__(0)
X_val = val_data[0]
Y_val = val_data[1]

## Training set
train_data = train_loader.__getitem__(0)
X_train = train_data[0]
Y_train = train_data[1]

## create an nfold training set. X,Y will be the baseline (1 fold) and X_train, Y_train Nfold
X_train_nf, Y_train_nf = train_loader_augmented.__getitem__(0)

for i in tqdm(range(Nfold-1)):
    X,Y = train_loader_augmented.__getitem__(0)
    X_train_nf = np.append(X_train_nf, X, axis=0)
    Y_train_nf = np.append(Y_train_nf, Y, axis=0)
    
