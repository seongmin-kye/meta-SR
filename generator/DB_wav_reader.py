import logging
import os
from glob import glob
import sys

# import librosa
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_feats(directory, pattern='**/*.pkl'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_feats_structure(directory, test=False):
    DB = pd.DataFrame()
    DB['filename'] = find_feats(directory) # filename
    DB['filename'] = DB['filename'].unique().tolist()
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6]) # dataset folder name

    speaker_list = sorted(set(DB['speaker_id']))  # len(speaker_list) == n_speakers
    if test: spk_to_idx = {spk: i+1211 for i, spk in enumerate(speaker_list)}
    else: spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name

    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB, len(DB), num_speakers