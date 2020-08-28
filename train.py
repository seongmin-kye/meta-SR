import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import warnings
import argparse

import configure as c
from str2bool import str2bool
from generator.DB_wav_reader import read_feats_structure
from generator.SR_Dataset import read_MFB_train as read_MFB

import torch
import torch.optim as optim

from model.model_normalize import background_resnet
from generator.meta_generator import metaGenerator
from losses.prototypical import Prototypical
from losses.softmax import SoftmaxLoss

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use cuda.')
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--data_type', type=str, default='vox2', help='vox1 or vox2.')

parser.add_argument('--loss_type', type=str, default='prototypical', help='prototypical or softmax.')
parser.add_argument('--use_GC', type=str2bool, default=True, help='Use global classification logit.')

max_epoch = 301
parser.add_argument('--use_checkpoint', type=str2bool, default=False, help='Use checkpoint.')
parser.add_argument('--cp_num', type=int, default=0, help='Number of checkpoint.')
# episode setting
parser.add_argument('--n_shot', type=int, default=1, help='Number of support set per class.')
parser.add_argument('--n_query', type=int, default=2, help='Number of queries per class.')
parser.add_argument('--use_variable', type=str2bool, default=True, help='Use variable query.')
parser.add_argument('--nb_class_train', type=int, default=100, help='Number of way for training episode.')
# random seed
parser.add_argument('--seed', type=int, default=100, help='Set random seed.')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
log_dir = 'saved_model/baseline_' + str(args.n_folder).zfill(3)  # where to save checkpoints

def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.use_checkpoint: start = args.cp_num + 1
    else: start = 0  # Start epoch
    n_epochs = max_epoch - start  # How many epochs?

    # Load dataset
    train_DB, n_data, n_classes = make_DB(DB_type=args.data_type)
    n_episode = int(n_data / ((args.n_shot + args.n_query) * args.nb_class_train))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate model and optimizer
    if args.use_checkpoint:
      model, optimizer = load_model(log_dir, args.cp_num, n_classes)
    else:
      model = background_resnet(num_classes=n_classes)
      optimizer = create_optimizer(model)

    # define objective function, optimizer and scheduler
    objective = Prototypical() if args.loss_type == 'prototypical' else SoftmaxLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, min_lr=1e-5, threshold=1e-4, verbose=1)

    if args.use_cuda:
        model.cuda()

    train_generator = metaGenerator(train_DB, read_MFB,
                                    nb_classes=args.nb_class_train, nb_samples_per_class=args.n_shot + args.n_query,
                                    max_iter=n_episode * (n_epochs-args.cp_num), xp=np)
    # training
    train(train_generator, model, objective, optimizer, n_episode, log_dir, scheduler)

def train(train_generator, model, objective, optimizer, n_episode, log_dir, scheduler):

    # switch to train mode
    model.train()

    # for batch_idx, (data) in enumerate(train_loader):
    log_interval = int(n_episode / 2)
    avg_train_losses = []
    for t, (data) in train_generator:
        epoch = int(t / n_episode) + args.cp_num

        if t % n_episode == 0:
            losses = AverageMeter()
            losses_e = AverageMeter()
            losses_g = AverageMeter()
            accuracy_e = AverageMeter()
            accuracy_g = AverageMeter()

        inputs, targets_g = data  # target size:(batch size,1), input size:(batch size, 1, dim, win)

        if args.loss_type == 'softmax':
            loss, acc_g = objective(inputs, targets_g, model)
            losses.update(loss.item(), inputs.size(0))
            accuracy_g.update(acc_g * 100, inputs.size(0))

            # episode number in epoch
            ith_episode = t % n_episode
            if ith_episode % log_interval == 0:
                print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Loss {loss.avg:.4f}\t'
                    'Acc {acc_global.avg:.4f}'.format(
                    epoch, ith_episode, n_episode, 100. * ith_episode / n_episode,
                    loss=losses, acc_global=accuracy_g))

        elif args.loss_type == 'prototypical':
            targets_e = tuple([i for i in range(args.nb_class_train)]) * (args.n_query)
            targets_e = torch.tensor(targets_e, dtype=torch.long).cuda(non_blocking=True)
            support, query = split_support_query(inputs)

            loss, loss_e, loss_g, acc_e, acc_g =\
                objective(support, query, targets_g, targets_e, model, args.use_GC)
            losses.update(loss.item(), query.size(0))
            losses_e.update(loss_e.item(), query.size(0))
            losses_g.update(loss_g.item(), inputs.size(0))
            accuracy_e.update(acc_e * 100, query.size(0))
            accuracy_g.update(acc_g * 100, inputs.size(0))

            # episode number in epoch
            ith_episode = t % n_episode
            if ith_episode % log_interval == 0:
                print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Loss {loss.avg:.4f} (loss_e: {loss_e.avg:.4f} / loss_g: {loss_g.avg:.4f})\t'
                    'Acc e / g {acc_episode.avg:.4f} / {acc_global.avg:.4f}'.format(
                    epoch, ith_episode, n_episode, 100. * ith_episode / n_episode,
                    loss=losses, loss_e=losses_e, loss_g=losses_g, acc_episode=accuracy_e, acc_global=accuracy_g))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % n_episode == 0 and t != 0: #epoch interval
            scheduler.step(losses.avg, epoch)

            # calculate average loss over an epoch
            avg_train_losses.append(losses.avg)

            # do checkpointing
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       '{}/checkpoint_{}.pth'.format(log_dir, str(epoch).zfill(3)))

    # find position of lowest training loss
    minposs = avg_train_losses.index(min(avg_train_losses)) + 1
    print('Lowest training loss at epoch %d' % minposs)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(log_dir, cp_num, n_classes):
    model = background_resnet(num_classes=n_classes)
    optimizer = create_optimizer(model)

    print('=> loading checkpoint')
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')

    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return model, optimizer

def create_optimizer(model, new_lr=1e-1, wd=1e-4):
    # setup optimizer
    optimizer = optim.SGD([
        {'params': model.parameters(), 'weight_decay': wd}
    ], lr=new_lr, momentum=0.9, nesterov=True, dampening=0)

    return optimizer

def make_DB(DB_type='vox2'):
    # Load training set
    data_dir = c.TRAIN_FEAT_DIR_2 if DB_type=='vox2' else c.TRAIN_FEAT_DIR_1
    train_DB, train_len, n_classes = read_feats_structure(data_dir)

    return train_DB, train_len, n_classes

def split_support_query(inputs):
    B, C, Fr, T = inputs.size()
    inputs = inputs.reshape(args.n_shot + args.n_query, args.nb_class_train, C, Fr, T)
    support = inputs[:args.n_shot].reshape(-1, C, Fr, T)
    query = inputs[args.n_shot:].reshape(-1, C, Fr, T)

    if args.use_variable:
        min_win, max_win = c.SHORT_SIZE, T
        win_size = random.randrange(min_win, max_win)
        query = query[:, :, :, :win_size].contiguous()

    return support, query

if __name__ == '__main__':
    main()