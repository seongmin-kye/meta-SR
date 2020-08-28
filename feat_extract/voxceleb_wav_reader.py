import logging
import os
from glob import glob
import sys
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_wavs(directory, pattern='**/*.wav'):
    """Recursively finds all waves matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def find_feats1(directory, pattern='*.pkl'):
    """Recursively finds all feats matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def find_feats2(directory, pattern='**/*.pkl'):
    """Recursively finds all feats matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


def read_voxceleb_structure(directory, data_type):
    voxceleb = pd.DataFrame()
    if data_type == 'wavs':
        voxceleb['filename'] = find_wavs(directory)
    elif data_type == 'feats':
        voxceleb['filename'] = find_feats1(directory)
    else:
        raise NotImplementedError
    voxceleb['filename'] = voxceleb['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    num_speakers = len(os.listdir(directory))
    logging.info('Found {} files with {} different speakers.'.format(str(len(voxceleb)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(voxceleb.head(10))
    return voxceleb