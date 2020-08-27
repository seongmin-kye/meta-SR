import numpy as np
import random
import torch
import torchvision.transforms as transforms
from generator.SR_Dataset import TruncatedInputfromMFB, ToTensorInput

class metaGenerator(object):

    def __init__(self, data_DB, file_loader, nb_classes=100, nb_samples_per_class=3,
                  max_iter=100, xp=np):
        super(metaGenerator, self).__init__()

        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(data_DB)
        self.file_loader = file_loader
        self.transform = transforms.Compose([
            TruncatedInputfromMFB(),  # numpy array:(1, n_frames, n_dims)
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

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.data.keys()))
        sampled_characters = random.sample(self.data.keys(), nb_classes)
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            label = picture_list[char]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(label, self.transform(self.file_loader(_imgs[i]))) for i in _ind])
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels