import copy
import os
import struct
import time
from random import shuffle

import numpy as np
from kivymd.toast import toast
from numpy.random import multinomial

MNIST_PATH = r'./data/mnist/MNIST/raw'
CIFAR_PATH = r'./data/cifar/CIFAR/raw'


class MNIST(object):
    def __init__(self, data_dir, train=None, shuffle=True, dimension=3):
        self.train = train
        self.train_x_set = None
        self.train_y_set = None
        self.train_labels_set = None

        self.test_x_set = None
        self.test_y_set = None
        self.test_labels_set = None

        self._shuffle = shuffle
        self.__dim = dimension
        if train is None or train is True:
            self.__load_mnist_train(data_dir)
        if train is None or train is False:
            self.__load_mnist_test(data_dir)

        self.__one_hot()
        self.__dimension()
        if self._shuffle:
            self.shuffle()
        self.__normalization()

    def __load_mnist_train(self, path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.train_x_set = images
        self.train_labels_set = labels

    def __load_mnist_test(self, path, kind='t10k'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.test_x_set = images
        self.test_labels_set = labels

    def __one_hot(self):
        if self.train in [None, True]:
            trn = np.zeros([len(self.train_labels_set), 10])
            for i, x in enumerate(self.train_labels_set):
                trn[i, x] = 1
            self.train_y_set = trn
        if self.train in [None, False]:
            te = np.zeros([len(self.test_labels_set), 10])
            for i, x in enumerate(self.test_labels_set):
                te[i, x] = 1
            self.test_y_set = te

    def __normalization(self, renormalize=False):
        mean = 0.1306604762738434
        std = 0.30810780385646314
        if self.train in [None, True]:
            self.train_x_set = self.train_x_set / 255.
            if self.__dim == 3 and renormalize is False:
                self.train_x_set -= mean
                self.train_x_set /= std

        if self.train in [None, False]:
            self.test_x_set = self.test_x_set / 255.
            if self.__dim == 3 and renormalize is False:
                self.test_x_set -= mean
                self.test_x_set /= std

        if self.__dim == 3 and renormalize is True:
            mean = 0
            std = 0
            for x in self.train_x_set:
                mean += np.mean(x[:, :, 0])
            mean /= len(self.train_x_set)
            self.train_x_set -= mean
            for x in self.train_x_set:
                std += np.mean(np.square(x[:, :, 0]).flatten())
            std = np.sqrt(std / len(self.train_x_set))
            print('The mean and std of MNIST:', mean, std)  # 0.1306604762738434 0.30810780385646314
            self.train_x_set /= std
            self.test_x_set -= mean
            self.test_x_set /= std

    def __dimension(self):
        if self.__dim == 1:
            pass
        elif self.__dim == 3:
            if self.train in [None, True]:
                self.train_x_set = np.reshape(self.train_x_set, [len(self.train_x_set), 28, 28, 1])
            if self.train in [None, False]:
                self.test_x_set = np.reshape(self.test_x_set, [len(self.test_x_set), 28, 28, 1])
        else:
            print('Dimension Error!')
            exit(1)

    def shuffle(self):
        if self.train in [None, True]:
            index = np.arange(len(self.train_x_set))
            np.random.shuffle(index)
            self.train_x_set = self.train_x_set[index]
            self.train_y_set = self.train_y_set[index]
            self.train_labels_set = self.train_labels_set[index]

        if self.train in [None, False]:
            index = np.arange(len(self.test_x_set))
            np.random.shuffle(index)
            self.test_x_set = self.test_x_set[index]
            self.test_y_set = self.test_y_set[index]
            self.test_labels_set = self.test_labels_set[index]

    @property
    def dataset(self):
        if self.train is None:
            return [self.train_x_set, self.test_x_set]
        elif self.train is True:
            return self.train_x_set
        elif self.train is False:
            return self.test_x_set

    @property
    def targets(self):
        if self.train is None:
            return [self.train_y_set, self.test_y_set]
        elif self.train is True:
            return self.train_y_set
        elif self.train is False:
            return self.test_y_set

    def __len__(self):
        if self.train is None:
            return len(self.train_x_set) + len(self.test_x_set)
        elif self.train is True:
            return len(self.train_x_set)
        elif self.train is False:
            return len(self.test_x_set)

    def __getitem__(self, index):
        if self.train is True:
            return self.train_x_set[index], self.train_y_set[index]
        elif self.train is False:
            return self.test_x_set[index], self.test_y_set[index]
        else:
            toast("Iterating MNIST dataset with Train=None is not supported")
            exit()

    def random_batches(self, batch_size, train=True, nb_batches=1):
        batches = []
        for i in range(nb_batches):
            ids = np.random.choice(range(len(self.train_x_set)), batch_size, replace=False)
            if self.train in [None, True] and train is True:
                batches.append((self.train_x_set[ids], self.train_y_set[ids]))
            elif self.train in [None, False] and train is False:
                batches.append((self.test_x_set[ids], self.test_y_set[ids]))
            else:
                raise "random_batches cannot return batches for train and test"
        return batches

    def slice(self, index):
        new_self = copy.deepcopy(self)
        if self.train in [None, True]:
            new_self.train_x_set = self.train_x_set[index]
            new_self.train_y_set = self.train_y_set[index]
        if self.train in [None, False]:
            new_self.test_x_set = self.test_x_set[index]
            new_self.test_y_set = self.test_y_set[index]

        return new_self

    def __str__(self):
        return f"MNIST/Train={self.train} with {len(self)} samples"


def inference_ds(peer, args):
    # todo handle inference scope
    if args.test_scope in ['global', 'neighborhood', 'local']:
        return peer.inference
    else:
        exit(f'Error: unrecognized TEST_SCOPE value: {args.test_scope}')
