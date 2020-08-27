import random
import pickle # For python3 
import numpy as np

import torch

import configure as c

def read_MFB_train(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)

    feature = feat_and_label['feat']  # size : (n_frames, dim=40)

    return feature

def read_MFB(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)
        
    feature = feat_and_label['feat'] # size : (n_frames, dim=40)
    label = feat_and_label['label']

    return feature, label

class TruncatedInputfromMFB(object):
    """
    input size : (n_frames, dim=40)
    output size : (1, n_win=40, dim=40) => one context window is chosen randomly
    """

    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        # Normalizing before slicing
        network_inputs = []
        num_frames = len(frames_features)
        win_size = c.NUM_WIN_SIZE
        half_win_size = int(win_size / 2)
        # if num_frames - half_win_size < half_win_size:
        while num_frames <= win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)

        for i in range(self.input_per_file):
            j = random.randrange(half_win_size, num_frames - half_win_size)
            if not j:
                frames_slice = np.zeros(num_frames, c.FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - half_win_size:j + half_win_size]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)

def normalize_frames(m,Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

class ToTensorInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            
            # input size : (1, n_win=200, dim=40)
            # output size : (1, dim=40, n_win=200)
            return ten_feature

class ToTensorTestInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            np_feature = np.expand_dims(np_feature, axis=0)
            np_feature = np.expand_dims(np_feature, axis=1)
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,1,3,2))).float() # output type => torch.FloatTensor, fast
            # input size : (1, 1, n_win=200, dim=40)
            # output size : (1, 1, dim=40, n_win=200)
            return ten_feature