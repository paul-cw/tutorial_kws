#!/usr/bin/env python
# coding: utf-8

# In[1]:


from config import *
from utility import *


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from os import fspath
from pathlib import Path

import librosa
import IPython.display as ipd
import random


# In[3]:


## fix the random seed
random.seed(0)
np.random.seed(0)


# ## download, unpack and prepare the dataset if not done yet

# In[4]:


print('try loading dataset from ', data_dir)


# In[5]:

import wget 

if not os.path.isdir(data_dir):
    print('loading dataset')
    os.mkdir(data_dir)
    print('start downloading')
    #os.system('wget -O %sspeech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'%data_dir)
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    to_name = "./data/speech_commands_v0.02.tar.gz"
    wget.download(url, out=to_name)
    print('start unzipping')
    os.system('tar -xf %sspeech_commands_v0.02.tar.gz -C '%data_dir + '%s'%data_dir[0:-1])
    print('start removing .zip file')
    os.system('rm %sspeech_commands_v0.02.tar.gz'%data_dir)
    print('done!')


# In[6]:


## create a 'silence' file for the 'silence' class
if not os.path.exists(data_dir + 'silence/silence.wav'):

    if not os.path.exists(data_dir + 'silence/'):
        os.mkdir(data_dir + 'silence/')
    
    import soundfile as sf
    sf.write(data_dir + 'silence/silence.wav', np.zeros(shape=(fs)), fs)


# In[7]:


## create a training_list.txt file
def get_file_content_as_list(fname):
    my_file = open(fname, "r")
    content = my_file.read()
    clist = content.split("\n")
    my_file.close()
    return clist

def create_training_list(validation_file, testing_file, data_dir, accepted_kws):
    """creates a training_list.txt file and returns the path where it saved it"""

    outputdir = data_dir + 'training_list.txt'

    if os.path.exists(outputdir):
        print('there is already a training_list.txt file, in', data_dir, 'abbording!')
        return outputdir

    vcontent = get_file_content_as_list(data_dir + validation_file)
    tcontent = get_file_content_as_list(data_dir + testing_file)

    wave_files_paths = [fspath(path) for path in Path(data_dir).rglob('*.wav')]
    tr_names = []
    for idx, wave_path in enumerate(wave_files_paths):
        lab = wave_path.split('/')[-2]
        fname = lab + '/' + wave_path.split('/')[-1]

        if fname not in vcontent and fname not in tcontent and lab in accepted_kws:
            tr_names.append(fname)

    f = open(outputdir, "w")
    for ele in tr_names:
        f.write(ele + '\n')
    f.close()

    return outputdir


# In[8]:


## create a training list file from the validation/testing files provided in the dataset
create_training_list('validation_list.txt', 'testing_list.txt', data_dir, accepted_kws=['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
                             'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
                             'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow',
                             'yes', 'zero', 'one'])

## check if it worked
pd.read_csv(data_dir + 'validation_list.txt')


# # create labels and metadata

# In[9]:


## config
silence_percentage= 0.1 # fraction numb of silence files added as fraction of total files (all files in kw), >0 !
unknown_percentage= 1.  # fraction numb of unknown files added as fraction of total files 1 -> take all


# In[10]:


## prepare the metadata info that are in the filenames into a pandas dataframe
df_all = []
tr_counter = 0

for se in ['testing', 'validation', 'training']:
    print('started processing ', se, 'set')
    list_path = data_dir + se + "_list.txt"
    
    ## read train test split lists
    df = pd.read_csv(list_path, header=None)

    ## extract keyword from filename
    df['keyword'] = df[0].apply(lambda x: x.split('/')[0])

    ## label the unwanted kws as 'unknown'
    df['keyword'] = df.keyword.apply(lambda x: x if x in kw else 'unknown')

    ## split into wanted keywords and unknown
    dfu = df[df.keyword =='unknown']
    df  = df[df.keyword != 'unknown']

    ## keep only some unknowns
    N    = int(len(df) * unknown_percentage)
    dfu = dfu.sample(n = N)

    ## create silence utterances
    dfs = pd.DataFrame()
    dfs[0] = ['silence/silence.wav'] * int(silence_percentage * len(df))
    dfs['keyword'] = ['silence'] * int(silence_percentage * len(df))

    ## combine 
    df = pd.concat([df,dfu,dfs])
    df = df.sample(frac=1, random_state=0)
    
    ## add dataset info
    if se == 'training':
        df['dataset'] = len(df) * [se]
        tr_counter +=1
    else:
        df['dataset'] = len(df) * [se]
        
    ## add metadata as columns
    df['speaker_info'] = df[0].apply(lambda x: x.split('/')[-1].split('.')[0] if 'silence' not in x else 'None')
    df['speaker_id'] = df['speaker_info'].apply(lambda x: x.split('_')[0]     if 'silence' not in x else 'None')
    df['speaker_ut'] = df['speaker_info'].apply(lambda x: x.split('_')[-1]    if 'silence' not in x else 'None')
    df['label'] = df['keyword'].apply( lambda x: np.where(np.array(kws_all)==x)[0][0])
    df['speaker_id'] = df.apply(lambda x: x['speaker_id'] if x['keyword']!= 'silence' else 
         df.speaker_id.unique()[np.random.randint(len(df.speaker_id.unique()),size=1)[0]],axis=1)
    df['label_one_hot'] = df.label.apply(lambda x: one_at(x, len(kws_all)))
    df = df.drop(['speaker_info','label'],axis=1)
    df_all.append(df)

df_all = pd.concat(df_all)

## give unique index to each row
df_all['idx'] = [i for i in range(len(df_all))]
df_all = df_all.set_index('idx')

## rename columns
df_all = df_all.rename(columns={0: "file_path"})


# In[11]:


df_all.to_pickle(data_dir + 'df_all.pkl')
print('saved metadata as df_all.pkl to ', data_dir)

