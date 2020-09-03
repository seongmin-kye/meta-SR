import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from generator.DB_wav_reader import read_feats_structure
from generator.SR_Dataset import read_MFB_train as read_MFB
from str2bool import str2bool
import configure as c

from model.model import background_resnet
from generator.meta_generator_test import metaGenerator

parser = argparse.ArgumentParser()
# Loading setting
parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use cuda.')
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--cp_num', type=int, default=100, help='Number of checkpoint.')
# Episode setting
parser.add_argument('--n_shot', type=int, default=1, help='Number of support set per class.')
parser.add_argument('--n_query', type=int, default=5, help='Number of queries per class.')
parser.add_argument('--nb_class_test', type=int, default=50, help='Number of way for test episode.')
parser.add_argument('--nb_episode', type=int, default=1000, help='Number of episode.')
# Test setting
parser.add_argument('--enroll_length', type=int, default=500, help='Length of enrollment utterance. (500=5s)')
parser.add_argument('--test_length', type=int, default=100, help='Length of test utterance. (100=1s)')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
log_dir = 'saved_model/baseline_' + str(args.n_folder).zfill(3)  # where to save checkpoints

def load_model(log_dir, cp_num, n_classes=5994):
    model = background_resnet(num_classes=n_classes)
    print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    model.load_state_dict(checkpoint['state_dict'])

    return model

def get_DB(feat_dir):
    DB = pd.DataFrame()
    for idx, i in enumerate(feat_dir):
        tmp_DB, _, _ = read_feats_structure(i, idx)
        DB = DB.append(tmp_DB, ignore_index=True)

    return DB

def evaluation(test_generator, model, use_cuda):

    total_acc = []
    ans_episode, n_episode = 0, 0
    log_interval = 100

    # switch to train mode
    model.eval()
    with torch.no_grad():
        # for batch_idx, (data) in enumerate(train_loader):
        for t, (data) in test_generator:
            inputs, targets_g = data  # target size:(batch size,1), input size:(batch size, 1, dim, win)
            support, query = inputs

            #normalize sliced input
            if c.USE_NORM:
                support = support - torch.mean(support, dim=3, keepdim=True)
                query = query - torch.mean(query, dim=3, keepdim=True)
            current_sample = query.size(0)  # batch size

            if use_cuda:
                support = support.cuda(non_blocking=True)
                query = query.cuda(non_blocking=True)

            targets_e = tuple([i for i in range(args.nb_class_test)]) * (args.n_query)
            targets_e = torch.tensor(targets_e, dtype=torch.long).cuda()

            support = model(support)  # out size:(batch size, #classes), for softmax
            query = model(query)

            support = support.reshape(args.n_shot, args.nb_class_test, -1)
            prototype = support.mean(dim=0)
            angle_e = F.linear(query, F.normalize(prototype))

            # calculate accuracy of predictions in the current batch
            temp_ans = (torch.max(angle_e, 1)[1].long().view(targets_e.size()) == targets_e).sum().item()
            total_acc.append(temp_ans/angle_e.size(0) * 100)

            ans_episode += temp_ans
            n_episode += current_sample
            acc_episode = 100. * ans_episode / n_episode

            if t % log_interval == 0:
                stds = np.std(total_acc, axis=0)
                ci95 = 1.96 * stds / np.sqrt(len(total_acc))
                print(('Accuracy_test {}-shot ={:.2f}({:.2f})').format(args.n_shot, acc_episode, ci95))

def main():
    # Load dataset
    feat_list = [c.TRAIN_FEAT_DIR_1, c.TEST_FEAT_DIR]
    test_DB = get_DB(feat_list)
    n_classes = 5994

    # print the experiment configuration
    print('\nNumber of classes (speakers) in test set:\n{}\n'.format(len(set(test_DB['labels']))))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = load_model(log_dir, args.cp_num, n_classes)
    if args.use_cuda:
        model.cuda()

    # make generator for unseen speaker identification
    test_generator = metaGenerator(test_DB, read_MFB, enroll_length=args.enroll_length, test_length=args.test_length,
                                   nb_classes=args.nb_class_test, n_support=args.n_shot, n_query=args.n_query,
                                   max_iter=args.nb_episode, xp=np)
    # evaluate
    evaluation(test_generator, model, args.use_cuda)

if __name__ == '__main__':
    main()