from __future__ import print_function
import os
import time
import argparse
import warnings
import pandas as pd

import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_curve

from str2bool import str2bool
from generator.SR_Dataset import *
from generator.DB_wav_reader import read_feats_structure
from model.model import background_resnet

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

parser = argparse.ArgumentParser()
# Loading setting
parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use cuda.')
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--cp_num', type=int, default=100, help='Number of checkpoint.')
parser.add_argument('--data_type', type=str, default='vox2', help='vox1 or vox2.')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
log_dir = 'saved_model/baseline_' + str(args.n_folder).zfill(3)

def main():
    # Load pair and test data
    veri_test_dir = 'lists/trial_pair_Verification.txt' #Original Vox1 test set
    test_feat_dir = [c.TEST_FEAT_DIR]
    test_DB = get_DB(test_feat_dir)
    n_classes = 5994 if args.data_type == 'vox2' else 1211

    # Load model from checkpoint
    model = load_model(args.use_cuda, log_dir, args.cp_num, n_classes)

    # Enroll and test
    tot_start = time.time()
    dict_embeddings = enroll_per_utt(test_DB, model)
    enroll_time = time.time() - tot_start

    # Perform verification
    verification_start = time.time()
    _ = perform_verification(veri_test_dir, dict_embeddings)
    tot_end = time.time()
    verification_time = tot_end - verification_start

    print("Time elapsed for enroll : %0.1fs" % enroll_time)
    print("Time elapsed for verification : %0.1fs" % verification_time)
    print("Total elapsed time : %0.1fs" % (tot_end - tot_start))


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
    # original saved file with DataParallel
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_d_vector(filename, model):
    input, label = test_input_load(filename)
    label = torch.tensor([1]).cuda()

    input = normalize_frames(input, Scale=c.USE_SCALE)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, 40, 40)
    input = Variable(input)
    with torch.no_grad():
        if args.use_cuda:
            #load gpu
            input = input.cuda()
            label = label.cuda()

        activation = model(input) #scoring function is cosine similarity so, you don't need to normalization

    return activation, label


def normalize_frames(m, Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def test_input_load(filename):
    feat_name = filename.replace('.wav', '.pkl')
    mod_filename = os.path.join(c.TEST_FEAT_DIR, feat_name)

    file_loader = read_MFB
    input, label = file_loader(mod_filename)  # input size :(n_frames, dim), label:'id10309'

    return input, label


def veri_test_parser(line):
    label = int(line.split(" ")[0])
    enroll_filename = line.split(" ")[1]
    test_filename = line.split(" ")[2].replace("\n", "")
    return label, enroll_filename, test_filename


def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    # print("EER: %0.5f, Threshold : %0.5f" %(eer, eer_threshold))
    print("Epoch=%d  EER= %.2f  Thres= %0.5f  DCF0.01= %.3f  DCF0.001= %.3f" % (
    args.cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold, np.min(DCF2), np.min(DCF3)))

    return eer, eer_threshold


def enroll_per_utt(test_DB, model):
    # Get enroll d-vector and test d-vector per utterance
    dict_embeddings = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key = os.sep.join(tmp_filename.split(os.sep)[-3:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key = os.path.splitext(key)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            dict_embeddings[key] = enroll_embedding
            print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))

    return dict_embeddings


def perform_verification(veri_test_dir, dict_embeddings):
    # Perform speaker verification using veri_test.txt
    f = open(veri_test_dir)
    score_list = []
    label_list = []
    num = 0

    while True:
        start = time.time()
        line = f.readline()
        if not line: break

        label, enroll_filename, test_filename = veri_test_parser(line)
        with torch.no_grad():
            # Get embeddings from dictionary
            enroll_embedding = dict_embeddings[enroll_filename]
            test_embedding = dict_embeddings[test_filename]

            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1
        end = time.time()
        print("%d) Score:%0.4f, Label:%s, Time:%0.4f" % (num, score, bool(label), end - start))

    f.close()
    eer, eer_threshold = get_eer(score_list, label_list)
    return eer


if __name__ == '__main__':
    main()