import numpy as np
import random

import torch
import torchvision.transforms as transforms

from generator.SR_Dataset import ToTensorInput
import configure as c

class metaGenerator(object):

    def __init__(self, test_DB, file_loader, enroll_length, test_length,
                 nb_classes=100, n_support=1, n_query=2, max_iter=100, xp=np):
        super(metaGenerator, self).__init__()

        self.nb_classes = nb_classes
        self.n_support = n_support
        self.n_query = n_query
        self.nb_samples_per_class = n_support+ n_query

        self.enroll_length = enroll_length
        self.test_length = test_length

        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.test_data = self._load_data(test_DB)
        self.file_loader = file_loader
        self.transform = transforms.Compose([
            ToTensorInput()  # torch tensor:(1, n_dims, n_frames)
        ])

    def _load_data(self, data_DB):
        nb_speaker = len(set(data_DB['labels']))

        return {key: np.array(data_DB.loc[data_DB['labels']==key]['filename']) for key in range(nb_speaker)}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()

    def cut_frames(self, frames_features, mode='enroll'):
        # Normalizing before slicing
        network_inputs = []
        num_frames = len(frames_features)

        if mode == 'enroll': win_size = self.enroll_length
        elif mode == 'test': win_size = self.test_length

        half_win_size = int(win_size / 2)
        # if num_frames - half_win_size < half_win_size:
        while num_frames <= win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)


        j = random.randrange(half_win_size, num_frames - half_win_size)
        if not j:
            frames_slice = np.zeros(num_frames, c.FILTER_BANK, 'float64')
            frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
        else:
            frames_slice = frames_features[j - half_win_size:j + half_win_size]
        network_inputs.append(frames_slice)

        return np.array(network_inputs)

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.test_data.keys()))
        sample_classes = random.sample(self.test_data.keys(), nb_classes)
        labels_and_images = []
        for (k, char) in enumerate(sample_classes):
            label = picture_list[char]
            # support(Enroll data) / query(Test data)
            data = self.test_data[char]
            _ind = random.sample(range(len(data)), nb_samples_per_class)
            # sample support
            labels_and_images.extend([(label, self.transform(self.cut_frames(self.file_loader(data[i]), mode='enroll'))) for i in _ind[:self.n_support]])
            # sample query
            labels_and_images.extend([(label, self.transform(self.cut_frames(self.file_loader(data[i]), mode='test'))) for i in _ind[self.n_support:]])

        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)

        support = torch.stack(images[:self.n_support * self.nb_classes], dim=0)
        query = torch.stack(images[self.n_support*self.nb_classes:], dim=0)

        labels = torch.tensor(labels, dtype=torch.long)

        return (support, query), labels

