from __future__ import print_function
import os
import argparse
import pandas as pd
from sklearn.metrics import roc_curve

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import torch.nn.functional as F
from torch.autograd import Variable

from generator.SR_Dataset import *
from str2bool import str2bool
from generator.DB_wav_reader import read_feats_structure
from model.model import background_resnet

# Training settings
parser = argparse.ArgumentParser()
# Loading setting
parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use cuda.')
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--cp_num', type=int, default=100, help='Number of checkpoint.')
# Test setting
parser.add_argument('--test_length', type=int, default=500, help='Length of test utterance. (100=1s)')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
log_dir = 'saved_model/baseline_' + str(args.n_folder).zfill(3)  # where to save checkpoints

def main():
    # Load model parameters
    log_dir = 'saved_model/baseline_'+str(args.n_folder).zfill(3)
    model = load_model(args.use_cuda, log_dir, args.cp_num, n_classes=5994)

    # Enroll and test
    test_feat_dir = [c.TRAIN_FEAT_DIR_1, c.TEST_FEAT_DIR] # use [train+test] set of VoxCeleb1
    # test_feat_dir = [c.TEST_FEAT_DIR]                   # use test set of VoxCeleb1
    test_DB = get_DB(test_feat_dir)

    # print the experiment configuration
    print('\nNumber of classes (speakers) in test set:\n{}\n'.format(len(set(test_DB['labels']))))

    eer, eer_threshold = enroll_and_verification(model, test_DB)

def get_DB(feat_dir):
    DB = pd.DataFrame()
    for idx, i in enumerate(feat_dir):
        tmp_DB, _, _ = read_feats_structure(i, idx)
        DB = DB.append(tmp_DB, ignore_index=True)
    return DB

def load_model(use_cuda, log_dir, cp_num, n_classes):
    model = background_resnet(num_classes=n_classes)

    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def get_d_vector(filename, model, mode='test'):
    input, label = read_MFB(filename)
    label = torch.tensor([1]).cuda()

    if mode == 'test':
        num_frames = len(input)
        win_size = args.test_length
        half_win_size = int(win_size / 2)

        # if num_frames - half_win_size < half_win_size:
        while num_frames <= win_size:
            input = np.append(input, input[:num_frames, :], axis=0)
            num_frames = len(input)

        j = random.randrange(half_win_size, num_frames - half_win_size)
        input = input[j - half_win_size:j + half_win_size]

    input = normalize_frames(input, Scale=c.USE_SCALE)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        if args.use_cuda:
            # load gpu
            input = input.cuda()
            label = label.cuda()
        activation = model(input)

    return activation, label

def normalize_frames(m,Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


def test_input_load(filename):
    input, label = read_MFB(filename)
    return input, label


def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    print("Epoch=%d  EER= %.2f  Thres= %0.5f  DCF0.01= %.3f  DCF0.001= %.3f" % (
    args.cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold, np.min(DCF2), np.min(DCF3)))

    return eer, eer_threshold


def enroll_and_verification(model, test_DB):
    """
    Get enroll d-vector and test d-vector per utterance.
    Perform speaker verification using veri_test.txt
    """
    score_list = []
    label_list = []
    num = 0
    nb_speaker = len(set(test_DB['labels']))
    for speaker in range(nb_speaker):
        """positive pair"""
        for i in range(100):
            label = 1
            pair_list = list(test_DB.loc[test_DB['labels'] == speaker]['filename'].sample(2))
            enroll_filename = pair_list[0]
            test_filename = pair_list[1]

            with torch.no_grad():
                enroll_embedding, enroll_label = get_d_vector(enroll_filename, model, mode='enroll')
                test_embedding, test_label = get_d_vector(test_filename, model, mode='test')

                score = F.cosine_similarity(enroll_embedding, test_embedding)
                score = score.data.cpu().numpy()[0]

                del enroll_embedding
                del test_embedding
            score_list.append(score)
            label_list.append(label)
            num += 1
            print("%d) Score:%0.4f, Label:%s" % (num, score, bool(label)))

        """nagative pair"""
        for i in range(100):
            label = 0
            enroll_filename = test_DB.loc[test_DB['labels'] == speaker]['filename'].sample(1).item()
            test_filename = test_DB.loc[test_DB['labels'] != speaker]['filename'].sample(1).item()

            with torch.no_grad():
                enroll_embedding, enroll_label = get_d_vector(enroll_filename, model, mode='enroll')
                test_embedding, test_label = get_d_vector(test_filename, model, mode='test')

                score = F.cosine_similarity(enroll_embedding, test_embedding)
                score = score.data.cpu().numpy()[0]

                del enroll_embedding
                del test_embedding
            score_list.append(score)
            label_list.append(label)
            num += 1
            print("%d) Score:%0.4f, Label:%s" % (num, score, bool(label)))

    eer, eer_threshold = get_eer(score_list, label_list)

    return eer, eer_threshold


if __name__ == '__main__':
    main()