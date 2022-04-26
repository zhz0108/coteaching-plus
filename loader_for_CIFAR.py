from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, root_dir, transform, mode, noise_file=''):

        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise for cifar10
        # generate asymmetric noise for cifar100
        self.transition_cifar100 = {}
        nb_superclasses = 20
        nb_subclasses = 5
        base = [1, 2, 3, 4, 0]
        for i in range(nb_superclasses * nb_subclasses):
            self.transition_cifar100[i] = int(base[i % 5] + 5 * int(i / 5))

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                # print(train_label)
                # print(len(train_label))
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            noise_label = json.load(open(noise_file, "r"))

            if self.mode == 'train':
                self.train_data = train_data
                self.noise_label = noise_label
                self.clean_label = train_label

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = cifar_dataset(dataset=self.dataset,
                                          root_dir=self.root_dir, transform=self.transform_train, mode="train",
                                          noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader, np.asarray(train_dataset.noise_label), np.asarray(train_dataset.clean_label)

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader


# test the custom loaders for CIFAR
dataset = 'cifar10'  # either cifar10 or cifar100
data_path = 'data/cifar-10-batches-py'  # path to the data file (don't forget to download the feature data and also put the noisy label file under this folder)

loader = cifar_dataloader(dataset,
                          batch_size=128,
                          num_workers=5,
                          root_dir=data_path,
                          noise_file='cifar10_noisy_labels_task1.json')

all_trainloader, noisy_labels, clean_labels = loader.run('train')
test_loader = loader.run('test')

# todo: Code your own algorithm
