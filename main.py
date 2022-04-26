# -*- coding:utf-8 -*-
from __future__ import print_function
from torch.utils.data import Dataset
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.newsgroups import NewsGroups
from data.torchlist import ImageFilelist
from model import MLPNet, CNN_small, CNN
import argparse, sys
import numpy as np
import datetime
import shutil
from PIL import Image
from loss import loss_coteaching, loss_coteaching_plus


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

            # self.train_noisy_labels = [i[0] for i in self.noise_label]
            # _train_labels = [i[0] for i in self.clean_label]
            self.noise_or_not = np.transpose(self.noise_label) == np.transpose(self.clean_label)

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
            return img, target, index

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
    parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
    parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, or imagenet_tiny', default='mnist')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
    parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Hyper Parameters
    batch_size = 128
    learning_rate = args.lr

    # load dataset
    if args.dataset == 'mnist':
        input_channel = 1
        init_epoch = 0
        num_classes = 10
        args.n_epoch = 200
        train_dataset = MNIST(root='./data/',
                              download=True,
                              train=True,
                              transform=transforms.ToTensor(),
                              noise_type=args.noise_type,
                              noise_rate=args.noise_rate
                              )

        test_dataset = MNIST(root='./data/',
                             download=True,
                             train=False,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate
                             )
    elif args.dataset == 'cifar10':
        input_channel = 3
        init_epoch = 20
        num_classes = 10
        args.n_epoch = 200
        train_dataset = cifar_dataset(dataset='cifar10',
                                      root_dir='data/cifar-10-batches-py',
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                      ]),
                                      mode="train",
                                      noise_file='cifar10_noisy_labels_task2.json')
        # CIFAR10(root='./data/',
        #                         download=True,
        #                         train=True,
        #                         transform=transforms.ToTensor(),
        #                         noise_type=args.noise_type,
        #                         noise_rate=args.noise_rate
        #                         )

        test_dataset = cifar_dataset(dataset='cifar10',
                                     root_dir='data/cifar-10-batches-py', transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]), mode='test')

        # CIFAR10(root='./data/',
        #         download=True,
        #         train=False,
        #         transform=transforms.ToTensor(),
        #         noise_type=args.noise_type,
        #         noise_rate=args.noise_rate
        #         )

    elif args.dataset == 'cifar100':
        input_channel = 3
        init_epoch = 5
        num_classes = 100
        args.n_epoch = 200
        train_dataset = cifar_dataset(dataset='cifar100',
                                      root_dir='data/cifar-100-python',
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                                      ]),
                                      mode="train",
                                      noise_file='cifar100_noisy_labels_task1.json')
        # CIFAR100(root='./data/',
        #                      download=True,
        #                      train=True,
        #                      transform=transforms.ToTensor(),
        #                      noise_type=args.noise_type,
        #                      noise_rate=args.noise_rate
        #                      )

        test_dataset = cifar_dataset(dataset='cifar100',
                                     root_dir='data/cifar-100-python',
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                                     ]),
                                     mode='test')
        # CIFAR100(root='./data/',
        #                     download=True,
        #                     train=False,
        #                     transform=transforms.ToTensor(),
        #                     noise_type=args.noise_type,
        #                     noise_rate=args.noise_rate
        #                     )

    if args.forget_rate is None:
        forget_rate = args.noise_rate
    else:
        forget_rate = args.forget_rate

    noise_or_not = train_dataset.noise_or_not

    # Adjust learning rate and betas for Adam Optimizer
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [learning_rate] * args.n_epoch
    beta1_plan = [mom1] * args.n_epoch
    for i in range(args.epoch_decay_start, args.n_epoch):
        alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
        beta1_plan[i] = mom2

    rate_schedule = gen_forget_rate(args, forget_rate, args.fr_type)

    save_dir = args.result_dir + args.dataset + '/' + args.model_type + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate)

    txtfile = save_dir + model_str + '.txt'
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    if args.dataset == 'mnist':
        clf1 = MLPNet()
    if args.dataset == 'cifar10':
        clf1 = CNN_small(num_classes)
    if args.dataset == 'cifar100':
        clf1 = CNN(n_outputs=num_classes)

    clf1.cuda()
    print(clf1.parameters)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    if args.dataset == 'mnist':
        clf2 = MLPNet()
    if args.dataset == 'cifar10':
        clf2 = CNN_small(num_classes)
    if args.dataset == 'cifar100':
        clf2 = CNN(n_outputs=num_classes)

    clf2.cuda()
    print(clf2.parameters)
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)

    with open(txtfile, 'a') as myfile:
        myfile.write('epoch train_acc1 train_acc2 test_acc1 test_acc2\n')

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    # evaluate models with random weights
    test_acc1, test_acc2 = evaluate(args, test_loader, clf1, clf2, model_str)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + " " + str(
            test_acc2) + "\n")

    data1 = open('data1.txt', 'w')
    data2 = open('data2.txt', 'w')

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch, alpha_plan, beta1_plan)
        adjust_learning_rate(optimizer2, epoch, alpha_plan, beta1_plan)

        train_acc1, train_acc2 = train(args, train_dataset, train_loader, epoch, init_epoch, batch_size, clf1,
                                       optimizer1, clf2, optimizer2, rate_schedule, noise_or_not, model_str)
        # evaluate models
        test_acc1, test_acc2 = evaluate(args, test_loader, clf1, clf2, model_str)
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        data1.write('Epoch [%d/%d] Model1 %.4f %% Model2 %.4f %%, avg %.4f %%\n' % (
            epoch + 1, args.n_epoch, test_acc1, test_acc2, (test_acc1 + test_acc2) / 2))
        data2.write('%.4f ' % ((test_acc1 + test_acc2) / 2))
        with open(txtfile, 'a') as myfile:
            myfile.write(
                str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + " " + str(
                    test_acc2) + "\n")

    data1.close()
    data2.close()


def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)


# define drop rate schedule
def gen_forget_rate(args, forget_rate, fr_type='type_1'):
    if fr_type == 'type_1':
        rate_schedule = np.ones(args.n_epoch) * forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    return rate_schedule


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(args, train_dataset, train_loader, epoch, init_epoch, batch_size, model1, optimizer1, model2, optimizer2,
          rate_schedule, noise_or_not, model_str):
    print('Training %s...' % model_str)

    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0

    for i, (data, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()

        labels = torch.tensor(Variable(labels).cuda(), dtype=torch.int64)

        data = Variable(data).cuda()

        # Forward + Backward + Optimize
        logits1 = model1(data)
        prec1, = accuracy(logits1, labels, topk=(1,))
        train_total += 1
        train_correct += prec1

        logits2 = model2(data)
        prec2, = accuracy(logits2, labels, topk=(1,))
        train_total2 += 1
        train_correct2 += prec2
        if epoch < init_epoch:
            loss_1, loss_2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        else:
            if args.model_type == 'coteaching_plus':
                loss_1, loss_2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], ind,
                                                            noise_or_not, epoch * i)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i + 1) % args.print_freq == 0:
            print(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                % (epoch + 1, args.n_epoch, i + 1, len(train_dataset) // batch_size, prec1, prec2, loss_1.item(),
                   loss_2.item()))

    train_acc1 = float(train_correct) / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total2)
    return train_acc1, train_acc2


# Evaluate the Model
def evaluate(args, test_loader, model1, model2, model_str):
    print('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2


if __name__ == '__main__':
    main()
