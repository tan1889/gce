import os.path
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from config import *
from random import randint
import sklearn.datasets
import zipfile
import shutil
import gzip
import csv
from warnings import warn
import random
import pickle


# Image dataset with heavy online transformation can slow down the training a lot. To solve this problem,
# one can increase the num_workers of the dataloader, or, even faster, pre-generate a lot of (e.g. 100x)
# transformations and store in files. Then mimic online transformation by randomly returning one of the
# transformed image of image[i] when it is queried. The following parameters are setting for this scenario.
MAX_EXPANSION = 100  # this many expansion for simulation of online image transformation
MAX_CACHEABLE_EXPANSION = 10  # if expansion > this, use DatasetFromExpandedTensor -> not cachable, but precalculated
EXPANSION_PER_FILE = 10  # how many times expansion per one files
TEST_RATIO = 0.2  # when test set is not specified, use this portion of the data as test set


# <editor-fold desc="COMMON METHODS AND UTILS ------------------------------------------------------------------------">


# parameters of some datasets, useful to determine some hyper-parameters automatically
ds_params = {'boston': {'n_train': 323, 'x_dim': 13, 'y_bound': 50, 'y_dim': 1},
             'diabetes': {'n_train': 282, 'x_dim': 10, 'y_bound': 332, 'y_dim': 1},
             'breast_cancer': {'n_train': 364, 'x_dim': 30, 'y_bound': 1, 'y_dim': 2},
             'digits': {'n_train': 1149, 'x_dim': 64, 'y_bound': 1, 'y_dim': 10},
             'iris': {'n_train': 96, 'x_dim': 4, 'y_bound': 1, 'y_dim': 3},
             'wine': {'n_train': 113, 'x_dim': 13, 'y_bound': 1, 'y_dim': 3},
             'housing': {'n_train': 13209, 'x_dim': 8, 'y_bound': 5, 'y_dim': 1},
             'msd': {'n_train': 370972, 'x_dim': 90, 'y_bound': 45, 'y_dim': 1},
             'cifar10_f': {'n_train': 40000, 'x_dim': 342, 'y_bound': 1, 'y_dim': 10},
             'mnist': {'n_train': 48000, 'x_dim': 784, 'y_bound': 1, 'y_dim': 10},
             'kddcup99': {'n_train': 3306440, 'x_dim': 41, 'y_bound': 1, 'y_dim': 23},
             'covertype': {'n_train': 11340, 'x_dim': 54, 'y_bound': 1, 'y_dim': 7},
             'miniboone': {'n_train': 83240, 'x_dim': 50, 'y_bound': 1, 'y_dim': 2},
             'susy': {'n_train': 3600000, 'x_dim': 18, 'y_bound': 1, 'y_dim': 2},
             'pufs': {'n_train': 4000000, 'x_dim': 128, 'y_bound': 1, 'y_dim': 2},
             'blogfeedback': {'n_train': 41917, 'x_dim': 280, 'y_bound': 1424, 'y_dim': 1},
             'transcode': {'n_train': 36684, 'x_dim': 25, 'y_bound': 215, 'y_dim': 1},
             }


def save_expanded_datasets(x, y, expand_func, filename):
    """save 100x expanded dataset to 5 files, each 20x of transformed dataset"""
    assert MAX_EXPANSION % EXPANSION_PER_FILE == 0, 'Invalid value for MAX_EXPANSION and EXPANSION_PER_FILE'
    n_files = MAX_EXPANSION // EXPANSION_PER_FILE
    file_path = filename.replace('.pt', '')
    for i in range(n_files):
        filename = '{}_train{}.pt'.format(file_path, i)
        keep_original = (i == 0)
        xt, yt = expand_func(x, y, EXPANSION_PER_FILE, keep_original)
        torch.save((xt, yt, EXPANSION_PER_FILE, y.shape[0]), filename)


def get_fixed_randperm(n):
    """return a fixed tensor of random permutation of n. Avoid setting manual_seed!"""
    perm = [i for i in range(n)]
    rg = random.Random(997)  # set a manual seed here so that the random permutation stays the same
    rg.shuffle(perm)
    return torch.LongTensor(perm)


def split_dataset(x, y, ratio):
    assert 0.05 <= ratio <= 0.95, 'ratio value ({}) is unacceptable'.format(ratio)
    n_rows = x.shape[0]
    indices = get_fixed_randperm(n_rows)
    split = int(n_rows * (1. - ratio))
    ids1 = indices[:split]
    ids2 = indices[split:]
    x_val = x[ids2]
    y_val = y[ids2]
    x_train = x[ids1]
    y_train = y[ids1]
    return x_train, y_train, x_val, y_val


def transform_and_apply_net(x, transform, net=None, cuda=False):
    batches = []
    i = 0
    while i < x.shape[0]:
        batch = []
        for j in range(i, i + 128):
            if j >= x.shape[0]:
                break
            b = transform(x[j])
            batch.append(b)
        batch = torch.stack(batch)
        if net:
            if cuda:
                batch = net(batch.cuda())
            else:
                batch = net(batch)

        batches.append(batch)
        i += 128
    new_x = torch.cat(batches).squeeze()
    return new_x.cpu()


def format_exp_valr(expansion, validation_ratio):
    expansion = max(1, int(expansion))
    validation_ratio = round(validation_ratio, 2)
    assert 0.05 <= validation_ratio <= 0.5 or validation_ratio == 0., 'Validation ratio value is not acceptable!'
    if validation_ratio == 0.: validation_ratio = 0.0  # for uniform filename later on
    return expansion, validation_ratio


def inform(verbose, s):
    if verbose >= 2:
        print('   ' + s)


def download_file(url_prefix, filename, path):
    import errno
    import socket
    from urllib.request import urlopen
    from urllib.error import URLError, HTTPError

    # create the directory if not exists
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    socket.timeout = 10
    file_path = os.path.join(path, filename)
    url = url_prefix + filename
    warn('Downloading dataset from ' + url)
    errmsg = ''
    for i in range(100):
        try:
            dl = urlopen(url)
            with open(file_path, 'wb') as f:
                f.write(dl.read())
        except (URLError, HTTPError) as error:
            errmsg = type(error).__name__
            if hasattr(error, 'reason'):
                errmsg = ' #{}: {}'.format(error.reason.errno, error.reason.strerror)
            warn('Download attempt #{}/100: Error: {}.\nRetrying...'.format(i+1, errmsg))
        except socket.timeout:
            errmsg = 'Error: Connection timeout'
            warn('Download attempt #{}/100: Error: {}.\nRetrying...'.format(i+1, errmsg))
        else:
            return file_path
    errmsg = 'Can not download dataset from {}. {}.'.format(url, errmsg)
    raise Exception(errmsg)


def normalize_data(x_train, x_test, x_val=None):
    """normalize input to zero mean one std assuming x_train, x_test are torch Tensors"""
    m = x_train.mean(0)
    s = x_train.std(0)
    x_train -= m
    x_test -= m
    if x_val: x_val -= m
    cols = []
    cols0 = []
    for i in range(s.size(0)):
        if s[i] > 1e-9:
            cols.append(i)
        else:
            cols0.append(i)

    if not cols0:  # no column has near zero std
        x_train /= s
        x_test /= s
        if x_val: x_val /= s
    elif cols:  # some columns have near zero std
        x_train[:, cols] /= s[cols]
        x_test[:, cols] /= s[cols]
        if x_val: x_val[:, cols] /= s[cols]
    if cols0:  # for columns with std ~ zero we just squash them
        if x_val:
            squash_data(x_train[:, cols0], x_test[:, cols0], x_val[:, cols0])
        else:
            squash_data(x_train[:, cols0], x_test[:, cols0])


def squash_data(x_train, x_test, x_val=None, a=0., b=1.):
    """squash input data in to range [a, b] assuming inputs are torch Tensors"""
    mn, _ = x_train.min(0)
    mx, _ = x_train.max(0)
    dv = (mx - mn) / (b - a)

    cols = []
    cols0 = []  # columns where dv=0 meaning all entries have a same value
    for i in range(dv.size(0)):
        if dv[i] > 0:
            cols.append(i)
        else:
            cols0.append(i)

    x_train -= mn
    x_test -= mn
    if x_val: x_val -= mn

    if not cols0:
        x_train /= dv
        x_train += a
        x_test /= dv
        x_test += a
        if x_val:
            x_val /= dv
            x_val += a
    elif cols:
        x_train[:, cols] /= dv[cols]
        x_train[:, cols] += a
        x_test[:, cols] /= dv[cols]
        x_test[:, cols] += a
        if x_val:
            x_val[:, cols] /= dv[cols]
            x_val[:, cols] += a

    if cols0:
        x_train[:, cols0] = 0  # min = max -> all entries are the same -> set to 0
        x_test[:, cols0] = 0
        if x_val: x_val[:, cols0] = 0


# </editor-fold>


# <editor-fold desc="CUSTOM DATASET AND DATALOADER -------------------------------------------------------------------">


class DatasetFromTensor(data.Dataset):

    def __init__(self, xs, ys, transform=None):  # xs, ys must be either numpy or torch.Tensor
        self.xs = xs
        self.ys = ys
        self._numpy_xs = self._numpy_ys = None
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.xs[index], self.ys[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.xs.shape[0]

    def numpy(self):
        if not self.transform:
            xs = self.xs.numpy() if torch.is_tensor(self.xs) else self.xs
            ys = self.ys.numpy() if torch.is_tensor(self.ys) else self.ys
        elif self._numpy_xs is not None and self._numpy_ys is not None:
            xs = self._numpy_xs
            ys = self._numpy_ys
        else:
            xs = []
            ys = []
            for i in range(self.ys.shape[0]):
                x, y = self[i]
                if torch.is_tensor(x):
                    x = x.numpy()
                if torch.is_tensor(y):
                    y = y.numpy()
                xs.append(x)
                ys.append(y)
            xs = np.stack(xs)
            ys = np.stack(ys)
            self._numpy_xs, self._numpy_ys = xs, ys
        return xs, ys


class DataLoader:

    def __init__(self, x, y, cuda, num_workers, batch_size=1, transform=None, shuffle=False, cacheable=True):
        # cacheable should be False if transform is random, i.e. data[i] returns different result each time requested
        self.cacheable = cacheable
        self.dataset = DatasetFromTensor(x, y, transform)

        self._kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
        if cuda: self._kwargs['pin_memory'] = True

        self._dataloader = torch.utils.data.DataLoader(self.dataset, **self._kwargs)

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self.dataset)

    def new(self, batch_size=None, shuffle=None):  # create a new data loader with different batch size
        # if (batch_size is None or self._kwargs['batch_size'] == batch_size) and \
        #         (shuffle is None or self._kwargs['shuffle'] == shuffle):
        #     # if parameters is the same, return the same dataloader, no need to create a new one
        #     # but this could affect multiprocessing when multiple processes uses a same dataloader
        #     return self
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.cacheable = self.cacheable
        cpy.dataset = self.dataset
        cpy._kwargs = self._kwargs
        if batch_size is not None:
            cpy._kwargs['batch_size'] = batch_size
        if shuffle is not None:
            cpy._kwargs['shuffle'] = shuffle
        cpy._dataloader = torch.utils.data.DataLoader(cpy.dataset, **cpy._kwargs)
        return cpy

    def subset(self, ids, batch_size=None, shuffle=None):  # create a new data loader for a subset of the dataset
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.cacheable = self.cacheable
        cpy.dataset = SubDataset(self.dataset, ids)
        cpy._kwargs = self._kwargs
        if batch_size is not None:
            cpy._kwargs['batch_size'] = batch_size
        if shuffle is not None:
            cpy._kwargs['shuffle'] = shuffle
        cpy._dataloader = torch.utils.data.DataLoader(cpy.dataset, **cpy._kwargs)
        return cpy

    @property
    def batch_size(self):
        return self._kwargs['batch_size']

    def numpy_data(self):
        return self.dataset.numpy()


class DatasetFromFiles(data.Dataset):
    """This class mimic online transformed dataset (e.g. mnist_t cifar10_t) using offline expanded dataset
    of 100x pre-transformed of original dataset. The dataset size is the same as original. But when
    item i is queried, the return value is one of the 100x transformed image of i stored in the expanded
    dataset including the original.
    Inputs: expansion defines the size of the dataset. E.g. expansion = 2 means len(this_dataset) = 2x
    len(original_dataset). However, query to each item i is still returned with one of the 100x randomly
    transformed instances of image i. So, it is intended that if expansion > 1, cacheable must be True,
    so the buffer will store a fixed version of the expanded dataset.
    Implementation problem: It seems dataloader reset all self.xx variable to its initial values after
    each epoch, so self.* is reset and we can not count the number of queries -> we can not load data
    dynamically. So, for now we load all data at once."""

    def __init__(self, file, expansion, transform):
        assert 1 <= expansion <= MAX_EXPANSION, 'expansion multiplier = {} is out of range'.format(expansion)
        self.file = file
        self.expansion = expansion
        self.transform = transform
        self._numpy_xs = self._numpy_ys = None

        self.total_expansion = 0
        self.original_size = None
        xs = []
        ys = []
        fid = 0
        fname = file.replace('.pt', '_train{}.pt'.format(fid))
        while os.path.exists(fname):
            assert os.path.getsize(fname) / 1e6 > 10., 'Expanded dataset file {} is corrupted!'.format(fname)
            x1, y1, expansion1, original_size1 = torch.load(fname)
            self.total_expansion += expansion1
            if self.original_size is None: self.original_size = original_size1
            assert self.original_size == original_size1, 'Mismatching dataset original size between data files!'
            fid += 1
            fname = file.replace('.pt', '_train{}.pt'.format(fid))
            xs.append(x1)
            ys.append(y1)
        assert self.total_expansion >= self.expansion, \
            'expansion ({}) is larger than total_expansion ({}) '.format(self.expansion, self.total_expansion)
        self.xs = torch.cat(xs) if torch.is_tensor(xs[0]) else np.concatenate(xs)
        self.ys = torch.cat(ys) if torch.is_tensor(ys[0]) else np.concatenate(ys)
        self.size = expansion * self.original_size

    def __getitem__(self, index):
        if self.expansion == 1:  # random transformation of image[index]
            segment = randint(0, self.total_expansion - 1)
            i = index + segment * self.original_size
            x, y = self.xs[i], self.ys[i]
        else:  # specific item
            x, y = self.xs[index], self.ys[index]

        if self.transform:  # should be a simple transform like normalization, otherwise no meaning for this class
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.size

    def numpy(self):
        if not self.transform:
            xs = self.xs.numpy() if torch.is_tensor(self.xs) else self.xs
            ys = self.ys.numpy() if torch.is_tensor(self.ys) else self.ys
        elif self._numpy_xs is not None and self._numpy_ys is not None:
            xs = self._numpy_xs
            ys = self._numpy_ys
        else:
            xs = []
            ys = []
            for i in range(self.ys.shape[0]):
                x, y = self[i]
                if torch.is_tensor(x):
                    x = x.numpy()
                if torch.is_tensor(y):
                    y = y.numpy()
                xs.append(x)
                ys.append(y)
            xs = np.stack(xs)
            ys = np.stack(ys)
            self._numpy_xs, self._numpy_ys = xs, ys
        return xs, ys
    

class DataLoaderFromFiles:

    def __init__(self, file, cuda, num_workers, expansion=1, batch_size=1, transform=None, shuffle=False):
        assert 0 <= expansion <= MAX_EXPANSION, 'expansion multiplier = {} is out of range'.format(expansion)
        self.cacheable = (expansion <= MAX_CACHEABLE_EXPANSION)  # small enough dataset so it still fit in memory
        if not self.cacheable or expansion == 0:  # simulate online transformation -> data size == original
            expansion = 1

        self._kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
        if cuda: self._kwargs['pin_memory'] = True

        self.dataset = DatasetFromFiles(file, expansion, transform)
        self._dataloader = torch.utils.data.DataLoader(self.dataset, **self._kwargs)

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self.dataset)
    
    def new(self, batch_size=None, shuffle=None):  # create a new data loader with different params
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.cacheable = self.cacheable
        cpy.dataset = self.dataset
        cpy._kwargs = self._kwargs
        if batch_size is not None:
            cpy._kwargs['batch_size'] = batch_size
        if shuffle is not None:
            cpy._kwargs['shuffle'] = shuffle
        cpy._dataloader = torch.utils.data.DataLoader(cpy.dataset, **cpy._kwargs)
        return cpy

    def subset(self, ids, batch_size=None, shuffle=None):
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.cacheable = self.cacheable
        cpy.dataset = SubDataset(self.dataset, ids)
        cpy._kwargs = self._kwargs
        if batch_size is not None:
            cpy._kwargs['batch_size'] = batch_size
        if shuffle is not None:
            cpy._kwargs['shuffle'] = shuffle
        cpy._dataloader = torch.utils.data.DataLoader(cpy.dataset, **cpy._kwargs)
        return cpy

    @property
    def batch_size(self):
        return self._kwargs['batch_size']

    def numpy_data(self):
        return self.dataset.numpy()


class SubDataset(data.Dataset):
    """This is for k-fold cross validation, we create a subset of a given dataset and reuse
    the underlying data tensors of the dataset so that no additional memory is created"""

    def __init__(self, dataset, ids):  # ids are indices of the subset of dataset
        self.ids = ids
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[self.ids[index]]

    def __len__(self):
        return self.ids.shape[0]

    def numpy(self):
        xs, ys = self.dataset.numpy()
        return xs[self.ids], ys[self.ids]


# </editor-fold>


# <editor-fold desc="CNN DATASETS: mnist, svhn, cifar10, cifar100 ----------------------------------------------------">


def normalization_info(x_train, x_test):  # print the normalization info
    ToTensor = transforms.Compose([transforms.ToTensor()])
    x = np.concatenate((x_train, x_test))
    l = []
    for i in range(x.shape[0]):
        l.append(ToTensor(x[i]))
    t = np.stack(l)
    for k in range(t.shape[1]):
        tk = t[:, k, :, :]
        print("channel {}: mean={:.3f} std={:.3f}".format(k, tk.mean(), tk.std()))


def get_cnn_transforms(dsname, crop, flip, totensor, normalize):
    """input is converted to 32x32 image first, see in CnnTransform, then this add the required transformations"""
    assert not normalize or totensor, 'Can only normalize when totensor is applied!'
    trfs = []
    size, padding = 32, 4
    if dsname == 'mnist':
        mean, std = [0.131], [0.308]
        trfs.append(transforms.Pad(2))  # increase mnist image size to 32x32, same as cifar, svhn
    elif dsname == 'svhn':
        mean, std = [0.442, 0.446, 0.472], [0.204, 0.208, 0.206]
    elif dsname == 'cifar2':
        mean, std = [0.499, 0.508, 0.520], [0.261, 0.259, 0.280]
    elif dsname == 'cifar10':
        mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    elif dsname == 'cifar100':
        mean, std = [0.507, 0.487, 0.441], [0.267, 0.257, 0.276]
    else:
        raise Exception("Unsupported CNN dataset '{}'".format(dsname))

    if crop:
        trfs.append(transforms.RandomCrop(size, padding=padding))
    if flip:
        trfs.append(transforms.RandomHorizontalFlip())
    if totensor:
        trfs.append(transforms.ToTensor())
        if normalize:
            trfs.append(transforms.Normalize(mean=mean, std=std))

    return trfs


class CnnTransform:

    def __init__(self, dsname, crop=False, flip=False, totensor=True, normalize=True):
        self.dsname = dsname
        self.transforms = get_cnn_transforms(dsname, crop, flip, totensor, normalize)

    def __call__(self, x):
        if self.dsname == "mnist":
            img = Image.fromarray(x, mode='L')
        else:
            img = Image.fromarray(x)
        for t in self.transforms:
            img = t(img)
        return img


def get_cnn_xy_data(path, dsname):
    if dsname == 'mnist':
        test_ds = datasets.MNIST(path, train=False, download=True)
        train_ds = datasets.MNIST(path, train=True, download=True)
    elif dsname == 'svhn':
        test_ds = datasets.SVHN(path, split='test', download=True)
        train_ds = datasets.SVHN(path, split='train', download=True)
    elif dsname == 'cifar10':
        test_ds = datasets.CIFAR10(path, train=False, download=True)
        train_ds = datasets.CIFAR10(path, train=True, download=True)
    elif dsname == 'cifar100':
        test_ds = datasets.CIFAR100(path, train=False, download=True)
        train_ds = datasets.CIFAR100(path, train=True, download=True)
    elif dsname == 'cifar2':
        x_train, y_train, x_test, y_test = pickle.load(open(path + 'cifar2.pkl', 'rb'))
        y_test = torch.LongTensor(y_test)
        y_train = torch.LongTensor(y_train)
        return x_train, y_train, x_test, y_test
    else:
        raise Exception("Unsupported dataset '{}'. Can only load MNIST, SVHN, CIFAR!".format(dsname))

    if hasattr(test_ds, 'data'):
        x_test = test_ds.data
        x_train = train_ds.data
    elif hasattr(test_ds, 'test_data'):
        x_test = test_ds.test_data
        x_train = train_ds.train_data
    else:
        raise Exception("Unsupported PyTorch version. Can not extract train/test set!")

    if hasattr(test_ds, 'targets'):
        y_test = torch.LongTensor(test_ds.targets)
        y_train = torch.LongTensor(train_ds.targets)
    elif hasattr(test_ds, 'labels'):
        y_test = torch.LongTensor(test_ds.labels)
        y_train = torch.LongTensor(train_ds.labels)
    elif hasattr(test_ds, 'test_labels'):
        y_test = torch.LongTensor(test_ds.test_labels)
        y_train = torch.LongTensor(train_ds.train_labels)
    else:
        raise Exception("Unsupported PyTorch version. Can not extract train/test set!")

    if dsname == "mnist":
        x_test = x_test.numpy()
        x_train = x_train.numpy()
    elif dsname == "svhn":
        x_test = x_test.transpose((0, 2, 3, 1))
        x_train = x_train.transpose((0, 2, 3, 1))

    return x_train, y_train, x_test, y_test


def load_cnn_dataset(dsname, cuda=False, num_workers=1, shuffle=True, validation_ratio=0, transform=True,
                     train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):
    _, validation_ratio = format_exp_valr(0, validation_ratio)

    x_train, y_train, x_test, y_test = get_cnn_xy_data(path, dsname)
    test_transform = CnnTransform(dsname, crop=False, flip=False)
    train_transform = CnnTransform(dsname, crop=transform, flip=transform and dsname.startswith("cifar"))

    if validation_ratio > 0:  # split train set to train and validation set
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:  # test set as validation set -> able to see test performance statistics during training
        x_val, y_val = x_test.copy(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size,
                              transform=train_transform, shuffle=shuffle, cacheable=False)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers,
                            batch_size=test_batch_size, transform=test_transform)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers,
                             batch_size=test_batch_size, transform=test_transform)

    regression = False
    x_shape = (1, 32, 32) if dsname == "mnist" else (3, 32, 32)
    n_classes = 10
    if dsname == "cifar2":
        n_classes = 2
    elif dsname == "cifar100":
        n_classes = 100

    return train_loader, val_loader, test_loader, x_shape, n_classes, regression


# </editor-fold>


# <editor-fold desc="MNIST DATASET -----------------------------------------------------------------------------------">


def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    totalImage = np.sum(image)
    m0 = np.sum(c0 * image) / totalImage
    m1 = np.sum(c1 * image)/totalImage
    m00 = np.sum((c0 - m0)**2 * image) / totalImage
    m11 = np.sum((c1 - m1)**2 * image) / totalImage
    m01 = np.sum((c0 - m0) * (c1-m1) * image) / totalImage
    mu_vector = np.array([m0, m1])
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix


def deskew(image):  # deskew image, only used for MNIST
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    center0 = np.array(image.shape) / 2.0
    offset = c-np.dot(affine, center0)
    return interpolation.affine_transform(image, affine, offset=offset)


def elastic_transform(image, alpha=34, sigma=4, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       alpha: intensity of deformation  (best: alpha = 34, sigma=4)
       sigma: scale of radom deformation, (8: similar to hand smoothness, 4: a lot of variety)
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def deskew_images(images):
    new_images = np.empty(images.shape, images.dtype)
    for i in range(images.shape[0]):
        new_images[i] = deskew(images[i])
    return new_images


def expand_mnist(images, labels, expansion=0, keep_original=True):
    n = images.shape[0]
    new_size = max(n, n * expansion)
    new_images = np.empty((new_size, images[0].shape[0], images[0].shape[1]), images.dtype)
    new_labels = np.empty(new_size, np.int64)
    i0 = 0
    if keep_original:
        i0 = n
        for i in range(n):  # add original images
            new_images[i] = images[i]
            new_labels[i] = labels[i]
    for i in range(i0, new_size):
        new_images[i] = elastic_transform(images[i % n])
        new_labels[i] = labels[i % n]
    return new_images, new_labels


def load_and_deskew_original_mnist(path, verbose):
    filename = os.path.join(path, 'mnist_deskew.pt')
    if os.path.exists(filename) and os.path.getsize(filename) / 1e6 > 10.:
        inform(verbose, 'Loading pre-calculated dataset from ' + filename)
        x_train, y_train, x_test, y_test = torch.load(filename)
    else:
        test_ds = datasets.MNIST(path, train=False, download=True)
        train_ds = datasets.MNIST(path, train=True, download=False)
        x_test, y_test = test_ds.test_data.numpy(), torch.LongTensor(test_ds.test_labels)
        x_train, y_train = train_ds.train_data.numpy(), torch.LongTensor(train_ds.train_labels)
        x_test = deskew_images(x_test)
        x_train = deskew_images(x_train)
        torch.save((x_train, y_train, x_test, y_test), filename)
    return x_train, y_train, x_test, y_test


def mnist_normalize(x): return (torch.from_numpy(x).float() - 33.386) / 78.654


def mnist_random_transform(x): return mnist_normalize(deskew(elastic_transform(x)))


def load_mnist(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, transform=False, expansion=0,
               train_batch_size=256, test_batch_size=256, verbose=2, path=config['DEFAULT']['datasets']):

    expansion, validation_ratio = format_exp_valr(expansion, validation_ratio)
    assert expansion < 2 or transform, \
        'expansion-ratio={} requires dataset transformation (name ending with _t)!'.format(expansion)

    filename = os.path.join(path, 'mnist_t_x{}_deskew_val{}.pt'.format(MAX_EXPANSION, validation_ratio))

    # load precomputed data for a specific case to load faster for experiments
    if expansion > 1 and os.path.isfile(filename) and os.path.getsize(filename) / 1e6 > 10.:
        inform(verbose, 'Loading pre-calculated dataset from ' + filename)
        x_val, y_val, x_test, y_test = torch.load(filename)
        val_loader = DataLoader(x_val, y_val, cuda, num_workers,
                                batch_size=test_batch_size, transform=mnist_normalize)
        test_loader = DataLoader(x_test, y_test, cuda, num_workers,
                                 batch_size=test_batch_size, transform=mnist_normalize)
        train_loader = DataLoaderFromFiles(filename, cuda, num_workers, expansion=expansion,
                                           batch_size=train_batch_size, transform=mnist_normalize, shuffle=shuffle)
    else:
        x_train, y_train, x_test, y_test = load_and_deskew_original_mnist(path, verbose)

        if validation_ratio > 0:  # split train set to train and validation set
            x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
        else:  # test set as validation set -> able to see test performance statistics during training
            x_val, y_val = x_test.copy(), y_test.clone()

        val_loader = DataLoader(x_val, y_val, cuda, num_workers,
                                batch_size=test_batch_size, transform=mnist_normalize)
        test_loader = DataLoader(x_test, y_test, cuda, num_workers,
                                 batch_size=test_batch_size, transform=mnist_normalize)

        if expansion <= 1:
            if transform:  # mnist with online transformation
                train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size,
                                          transform=mnist_random_transform, shuffle=shuffle, cacheable=False)
            else:  # original mnist deskew dataset
                train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size,
                                          transform=mnist_normalize, shuffle=shuffle, cacheable=True)
        else:  # transform=True! enlarge training set using elastic transformation
            inform(verbose, 'Computing {}x expansion of mnist_t with val={}'.format(MAX_EXPANSION, validation_ratio))
            save_expanded_datasets(x_train, y_train, expand_mnist, filename)
            with open(filename, 'wb') as f_out:
                torch.save((x_val, y_val, x_test, y_test), f_out)
            inform(verbose, 'Pre-computed dataset saved to ' + filename)
            train_loader = DataLoaderFromFiles(filename, cuda, num_workers, expansion=expansion,
                                               batch_size=train_batch_size, transform=mnist_normalize, shuffle=shuffle)

    x_shape, n_classes, regression = (1, 28, 28), 10, False
    return train_loader, val_loader, test_loader, x_shape, n_classes, regression


# </editor-fold>


# <editor-fold desc="CIFAR10 DATASET ---------------------------------------------------------------------------------">


def load_cifar10_data(path):
    test_ds = datasets.CIFAR10(path, train=False, download=True)
    train_ds = datasets.CIFAR10(path, train=True, download=True)

    if torch.__version__.startswith('0.') or torch.__version__.startswith('1.0.0'):
        x_test = test_ds.test_data
        y_test = torch.LongTensor(test_ds.test_labels)
        x_train = train_ds.train_data
        y_train = torch.LongTensor(train_ds.train_labels)
    else:
        x_test = test_ds.data
        y_test = torch.LongTensor(test_ds.targets)
        x_train = train_ds.data
        y_train = torch.LongTensor(train_ds.targets)
    return x_train, y_train, x_test, y_test


cfr10_cropflipnorm = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

cfr10_normalize = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])


def cfr10_random_transform(x): return cfr10_cropflipnorm(Image.fromarray(x))


def expand_cifar10(images, labels, expansion, keep_original, use_features_net):
    net = None
    cuda = False
    if use_features_net:
        from models.features import Cifar10F_342x1x1
        net = Cifar10F_342x1x1()
        net.load_state_dict(torch.load('features_nets/cifar10_342f.pt'))
        for param in net.parameters():
            param.requires_grad = False
        net.eval()
        if torch.cuda.is_available():
            net.cuda()
            cuda = True

    new_images = []
    new_labels = []
    i0 = 0
    if keep_original:
        i0 = 1
        new_images.append(transform_and_apply_net(images, cfr10_normalize, net, cuda))
        new_labels.append(np.int64(labels))
    for i in range(i0, expansion):
        new_images.append(transform_and_apply_net(images, cfr10_random_transform, net, cuda))
        new_labels.append(np.int64(labels))
    new_images = torch.cat(new_images)
    new_labels = torch.from_numpy(np.concatenate(new_labels))
    return new_images, new_labels


def expand_cifar10f(images, labels, expansion=1, keep_original=True):
    return expand_cifar10(images, labels, expansion, keep_original, use_features_net=True)


def expand_cifar10t(images, labels, expansion=1, keep_original=True):
    return expand_cifar10(images, labels, expansion, keep_original, use_features_net=False)


def load_cifar10f(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, expansion=0,
                  train_batch_size=256, test_batch_size=256, verbose=2, path=config['DEFAULT']['datasets']):

    expansion, validation_ratio = format_exp_valr(expansion, validation_ratio)

    filename1 = os.path.join(path, 'cifar10_f.pt')
    filename = os.path.join(path, 'cifar10_f_x{}_val{}.pt'.format(MAX_EXPANSION, validation_ratio))

    if expansion > 1 and os.path.isfile(filename) and os.path.getsize(filename) / 1e6 > 10.:
        inform(verbose, 'Loading pre-computed dataset from ' + filename)
        x_val, y_val, x_test, y_test = torch.load(filename)
        train_loader = DataLoaderFromFiles(filename, cuda, num_workers, expansion=expansion,
                                           batch_size=train_batch_size, shuffle=shuffle)
        normalize_data(train_loader.dataset.xs, x_test, x_val)
    elif expansion == 1 and os.path.isfile(filename1) and os.path.getsize(filename1) / 1e6 > 10.:
        x_train, y_train, x_test, y_test = torch.load(filename1)
        normalize_data(x_train, x_test)
        if validation_ratio > 0:  # split train set to train and validation set
            x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
        else:  # test set as validation set -> able to see test performance statistics during training
            x_val, y_val = x_test.copy(), y_test.copy()
        train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)

    else:
        x_train, y_train, x_test, y_test = load_cifar10_data(path)

        if expansion <= 1:
            x_test, y_test = expand_cifar10f(x_test, y_test, expansion=0)
            x_train, y_train = expand_cifar10f(x_train, y_train, expansion=0)
            torch.save((x_train, y_train, x_test, y_test), filename1)
            inform(verbose, 'Pre-computed dataset saved to ' + filename1)
            normalize_data(x_train, x_test)
            if validation_ratio > 0:  # split train set to train and validation set
                x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
            else:  # test set as validation set -> able to see test performance statistics during training
                x_val, y_val = x_test.copy(), y_test.clone()
            train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
        else:
            if validation_ratio > 0:  # split train set to train and validation set
                x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
            else:  # test set as validation set -> able to see test performance statistics during training
                x_val, y_val = x_test.copy(), y_test.clone()

            x_val, y_val = expand_cifar10f(x_val, y_val, expansion=0)
            x_test, y_test = expand_cifar10f(x_test, y_test, expansion=0)

            inform(verbose, 'Computing {}x expansion of cifar10_f with val={}'.format(MAX_EXPANSION, validation_ratio))
            save_expanded_datasets(x_train, y_train, expand_cifar10f, filename)
            torch.save((x_val, y_val, x_test, y_test), filename)
            inform(verbose, 'Pre-computed dataset saved to ' + filename)
            train_loader = DataLoaderFromFiles(filename, cuda, num_workers, expansion=expansion,
                                               batch_size=train_batch_size, shuffle=shuffle)
            normalize_data(train_loader.dataset.xs, x_test, x_val)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size)

    x_shape, n_classes, regression = x_test.shape[1:], 10, False
    return train_loader, val_loader, test_loader, x_shape, n_classes, regression


def load_cifar10(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, transform=False, expansion=0,
                 train_batch_size=256, test_batch_size=256, verbose=2, path=config['DEFAULT']['datasets']):

    expansion, validation_ratio = format_exp_valr(expansion, validation_ratio)
    assert expansion < 2 or transform, \
        'expansion-ratio={} requires dataset transformation (name ending with _t)!'.format(expansion)

    filename = os.path.join(path, 'cifar10_t_x{}_val{}.pt'.format(MAX_EXPANSION, validation_ratio))

    if expansion > 1 and os.path.isfile(filename) and os.path.getsize(filename) / 1e6 > 10.:
        inform(verbose, 'Loading pre-calculated data from ' + filename)
        x_val, y_val, x_test, y_test = torch.load(filename)
        val_loader = DataLoader(x_val, y_val, cuda, num_workers,
                                batch_size=test_batch_size, transform=cfr10_normalize)
        test_loader = DataLoader(x_test, y_test, cuda, num_workers,
                                 batch_size=test_batch_size, transform=cfr10_normalize)
        train_loader = DataLoaderFromFiles(filename, cuda, num_workers, expansion=expansion,
                                           batch_size=train_batch_size, transform=cfr10_normalize, shuffle=shuffle)
    else:
        x_train, y_train, x_test, y_test = load_cifar10_data(path)

        if validation_ratio > 0:  # split train set to train and validation set
            x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
        else:  # test set as validation set -> able to see test performance statistics during training
            x_val, y_val = x_test.copy(), y_test.clone()

        val_loader = DataLoader(x_val, y_val, cuda, num_workers,
                                batch_size=test_batch_size, transform=cfr10_normalize)
        test_loader = DataLoader(x_test, y_test, cuda, num_workers,
                                 batch_size=test_batch_size, transform=cfr10_normalize)

        if expansion <= 1:
            if transform:  # mnist with online transformation
                train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size,
                                          transform=cfr10_random_transform, shuffle=shuffle, cacheable=False)
            else:  # original mnist deskew dataset
                train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size,
                                          transform=cfr10_normalize, shuffle=shuffle, cacheable=True)
        else:  # transform=True! enlarge training set using elastic transformation
            inform(verbose, 'Computing {}x expansion of cifar10_t with val={}'.format(MAX_EXPANSION, validation_ratio))
            save_expanded_datasets(x_train, y_train, expand_cifar10t, filename)
            with open(filename, 'wb') as f_out:
                torch.save((x_val, y_val, x_test, y_test), f_out)
            inform(verbose, 'Pre-computed dataset saved to ' + filename)
            train_loader = DataLoaderFromFiles(filename, cuda, num_workers, expansion=expansion,
                                               batch_size=train_batch_size, transform=cfr10_normalize, shuffle=shuffle)

    x_shape, n_classes, regression = (3, 32, 32), 10, False
    return train_loader, val_loader, test_loader, x_shape, n_classes, regression


# </editor-fold>


# <editor-fold desc="MSD DATASET: Song release year prediction based on sound attributes------------------------------">


def download_extract_msd(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
    filename = "YearPredictionMSD.txt.zip"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 200.:
        # download if file not exists or smaller than 200MB -> appears broken
        file_path = download_file(url, filename, path)

    with zipfile.ZipFile(file_path) as myzip:
        myzip.extractall(path)

    # process and save as torch files
    warn('Downloading and preparing MSD (year prediction) dataset...')
    xs = []
    ys = []
    file_path = file_path.replace('.zip', '')
    with open(file_path, 'r') as f:
        for line in f:
            r = line.split(',')
            ys.append(r[0])
            xs.append(r[1:])
    os.unlink(file_path)

    x = torch.from_numpy(np.array(xs, float)).float()
    y = torch.from_numpy(np.array(ys, int)).int()

    split_id = 463715  # this is where the dataset split into train / test set
    x_train = x[:split_id]
    x_test = x[split_id:]
    y_train = y[:split_id]
    y_test = y[split_id:]

    file_path = file_path.replace('.txt', '.pt')
    torch.save((x_train, y_train, x_test, y_test), file_path)


def load_msd(cuda=False, num_workers=1, shuffle=True, validation_ratio=0,
             preprocessing='normalize, y-middle', train_batch_size=256, test_batch_size=256,
             path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'YearPredictionMSD.pt')):
        download_extract_msd(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'YearPredictionMSD.pt')
    y_train = y_train.long()
    y_test = y_test.long()

    x_shape, y_shape, regression = list(x_train.shape[1:]), 1, True  # number of features / output dim

    k = 12  # number of timbres

    if 'y-classes' in preprocessing:
        y_train -= 1922  # y becomes 0...89: 90 classes
        y_test -= 1922
        y_shape = 90
    elif 'y-mean' in preprocessing or 'y-middle' in preprocessing:
        # center years around mean=1998 or median=1967, range is [1922 - 2011]
        y_mean = 1998 if 'y-mean' in preprocessing else 1967
        y_train -= y_mean
        y_test -= y_mean

    if 'normalize' in preprocessing:
        m = torch.mean(x_train[:, :k], dim=0)  # means of averages of timbres (first k features) in training data
        s = torch.std(x_train[:, :k], dim=0)  # standard deviations of averages of timbres (first k features)
        x_train[:, :k] = (x_train[:, :k] - m) / s  # whitening 12 timbres averages features
        x_test[:, :k] = (x_test[:, :k] - m) / s
        # because we scaled the averages by 1/s, we need to scale the next 12 features (variances) by 1/s^2
        x_train[:, k:2*k] = x_train[:, k:2*k] / s**2
        x_test[:, k:2*k] = x_test[:, k:2*k] / s**2
        # and scale the next features (covariances) by 1/(s_i*s_j)
        col = 23
        for i in range(k):
            for j in range(i + 1, k):
                col += 1
                x_train[:, col] /= s[i] * s[j]
                x_test[:, col] /= s[i] * s[j]
    elif 'squash' in preprocessing:  # squash data range to [-1 1], ignoring meaning of features, probably not good
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_shape, y_shape, regression


# </editor-fold>


# <editor-fold desc="COVERTYPE DATASET: Forest Types Prediction ------------------------------------------------------">


def download_extract_covertype(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/"
    filename = "covtype.data.gz"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 10.:
        # download if file not exists or smaller than 10MB -> appears broken
        file_path = download_file(url, filename, path)

    # process and save as torch files
    warn('Downloading and preparing CoverType dataset...')
    xs = torch.FloatTensor(581012, 54)
    ys = torch.IntTensor(581012)
    i = -1
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            i += 1
            line = line.decode().replace('\n', '')
            r = line.split(',')
            ys[i] = int(r[-1]) - 1  # 0-based instead of 1-based
            xs[i] = torch.from_numpy(np.array(r[:-1], int)).float()

    split_id = 15120  # this is where the dataset split into train / test set
    x_train = xs[:split_id]
    x_test = xs[split_id:]
    y_train = ys[:split_id]
    y_test = ys[split_id:]

    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'CoverType.pt'))


def load_covertype(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, preprocessing='normalize',
                   train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'CoverType.pt')):
        download_extract_covertype(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'CoverType.pt')
    y_train = y_train.long()
    y_test = y_test.long()

    if 'normalize' in preprocessing:
        # means and stds of quantitative features, col 10-54 are bit indicators -> leave as it
        m = torch.Tensor([2959.36, 155.65, 14.10, 269.43, 46.42, 2350.15, 212.15, 223.32, 142.53, 1980.29])
        s = torch.Tensor([279.98, 111.91, 7.49, 212.55, 58.30, 1559.25, 26.77, 19.77, 38.27, 1324.19])
        x_train[:, :10] -= m
        x_train[:, :10] /= s
        x_test[:, :10] -= m
        x_test[:, :10] /= s
    elif 'squash' in preprocessing:  # squash data range to [-1 1]
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_val = x_train[11340:]
        y_val = y_train[11340:]
        x_train = x_train[:11340]
        y_train = y_train[:11340]
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    x_shape, n_classes, regression = list(x_train.shape[1:]), 7, False  # number of features / output dim
    return train_loader, val_loader, test_loader, x_shape, n_classes, regression


# </editor-fold>


# <editor-fold desc="SUSY DATASET: Particle Detection ----------------------------------------------------------------">


def download_extract_susy(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00279/"
    filename = "SUSY.csv.gz"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 870.:
        # download if file not exists or smaller than 870MB -> appears broken
        file_path = download_file(url, filename, path)

    # process and save as torch files
    warn('Downloading and preparing SUSY dataset...')
    xs = torch.FloatTensor(5000000, 18)
    ys = torch.IntTensor(5000000)
    i = -1
    with gzip.open(file_path, 'r') as f:
        for line in f:
            i += 1
            line = line.decode().replace('\n', '')
            r = line.split(',')
            ys[i] = int(float(r[0]))  # class
            xs[i] = torch.from_numpy(np.array(r[1:], float)).float()

    assert i == 5000000 - 1, 'Expected 5000000 data rows, got {} rows'.format(i + 1)
    test_id = 500000  # this is where the dataset split into train / test set
    x_train = xs[:-test_id]
    y_train = ys[:-test_id]
    x_test = xs[-test_id:]
    y_test = ys[-test_id:]

    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'SUSY.pt'))


def load_susy(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, preprocessing='normalize',
              train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'SUSY.pt')):
        download_extract_susy(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'SUSY.pt')
    y_train, y_test = y_train.long(), y_test.long()
    x_shape = list(x_train.shape[1:])

    if 'regression' in preprocessing:
        y_train[y_train <= 0] = -1
        y_test[y_test <= 0] = -1
        y_shape = 1
    else:
        y_shape = 2

    if 'normalize' in preprocessing:
        normalize_data(x_train, x_test)
    elif 'squash' in preprocessing:  # squash data range to [-1 1]
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_shape, y_shape, 'regression' in preprocessing


# </editor-fold>


# <editor-fold desc="PUFs DATASET: IoT security attacks detection ----------------------------------------------------">


def download_extract_pufs(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00463/"
    filename = "XOR_Arbiter_PUFs.zip"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 150.:
        # download if file not exists or smaller than 150MB -> appears broken
        file_path = download_file(url, filename, path)

    with zipfile.ZipFile(file_path) as myzip:
        myzip.extractall(path)

    # process and save as torch files
    warn('Downloading and preparing PUFs dataset...')
    file_path = file_path.replace('.zip', '')

    i = -1
    x_train = torch.CharTensor(5000000, 128)
    y_train = torch.CharTensor(5000000)
    with open(os.path.join(file_path, '5xor_128bit/train_5xor_128dim.csv'), 'r') as f:
        for line in f:
            i += 1
            r = line.split(',')
            x_train[i] = torch.from_numpy(np.array(r[:-1], np.uint8))
            y_train[i] = int(r[-1])

    assert i == 5000000 - 1, 'Expected 5 000 000 train data rows, got {} rows'.format(i + 1)

    i = -1
    x_test = torch.CharTensor(1000000, 128)
    y_test = torch.CharTensor(1000000)
    with open(os.path.join(file_path, '5xor_128bit/test_5xor_128dim.csv'), 'r') as f:
        for line in f:
            i += 1
            r = line.split(',')
            x_test[i] = torch.from_numpy(np.array(r[:-1], np.uint8))
            y_test[i] = int(r[-1])

    assert i == 1000000 - 1, 'Expected 1 000 000 test data rows, got {} rows'.format(i + 1)

    shutil.rmtree(file_path)
    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'PUFs.pt'))


def load_pufs(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, preprocessing='normalize',
              train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'PUFs.pt')):
        download_extract_pufs(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'PUFs.pt')
    x_train, x_test = x_train.float(), x_test.float()
    y_train, y_test = y_train.long(), y_test.long()
    x_shape = list(x_train.shape[1:])

    if 'regression' in preprocessing:
        y_shape = 1
    else:  # classification, change y to {0, 1}
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        y_shape = 2

    # pufs features are either -1 or 1 -> no need to be normalized

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_shape, y_shape, 'regression' in preprocessing


# </editor-fold>


# <editor-fold desc="TRANSCODING DATASET: Prediction of video transcoding time based on params -----------------------">


def download_extract_transcode(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00335/"
    filename = "online_video_dataset.zip"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 14.:
        # download if file not exists or smaller than 14MB -> appears broken
        file_path = download_file(url, filename, path)

    with zipfile.ZipFile(file_path) as myzip:
        myzip.extractall(path)

    # process and save as torch files
    warn('Downloading and preparing Transcode dataset...')
    file_path = file_path.replace('online_video_dataset.zip', '')

    i = -1
    n_rows = 68784
    num_features = 17  # number of numerical (quantitative) features

    x = torch.zeros(n_rows, num_features + 8)
    y = torch.FloatTensor(n_rows)

    codec_map = {'mpeg4': 0, 'h264': 1, 'vp8': 2, 'flv': 3}
    # ignoring id (not an attribute) and b_size (always zero)
    key_map = {'id': 1000000, 'duration': 0, 'codec': 100, 'width': 1, 'height': 2,
               'bitrate': 3, 'framerate': 4, 'i': 5, 'p': 6, 'b': 7, 'frames': 8,
               'i_size': 9, 'p_size': 10, 'b_size': 1000000, 'size': 11, 'o_codec': 101,
               'o_bitrate': 12, 'o_framerate': 13, 'o_width': 14, 'o_height': 15,
               'umem': 16, 'utime': 1000}

    with open(os.path.join(file_path, 'transcoding_mesurment.tsv'), 'r') as f:
        reader = csv.DictReader(f, dialect='excel-tab')
        for r in reader:
            i += 1
            for k, v in r.items():
                j = key_map[k]
                if j < 20:  # numerical attributes
                    x[i, j] = float(v)
                elif j == 100:  # codec = first categorical attribute -> 4 bit encoding
                    j = num_features + codec_map[v]
                    x[i, j] = 1
                elif j == 101:  # o_codec = second categorical attribute -> next 4 bit encoding
                    j = num_features + 4 + codec_map[v]
                    x[i, j] = 1
                elif j == 1000:
                    y[i] = float(v)

    assert i == n_rows - 1, 'Expected 68 784 data rows, got {} rows'.format(i + 1)
    os.unlink(os.path.join(file_path, 'transcoding_mesurment.tsv'))
    os.unlink(os.path.join(file_path, 'youtube_videos.tsv'))
    os.unlink(os.path.join(file_path, 'README.txt'))

    rids = torch.randperm(x.shape[0])  # shuffle the rows
    x = x[rids]
    y = y[rids]

    train_id = int((2./3.) * n_rows)
    x_train = x[:train_id]
    y_train = y[:train_id]
    x_test = x[train_id:]
    y_test = y[train_id:]

    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'Transcode.pt'))


def load_transcode(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, preprocessing='normalize',
                   train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'Transcode.pt')):
        download_extract_transcode(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'Transcode.pt')

    if 'normalize' in preprocessing:
        # means and stds of quantitative features of the train set, col 18-26 are bit indicators -> leave as it
        normalize_data(x_train[:, :17], x_test[:, :17])
    elif 'squash' in preprocessing:  # squash data range to [-1 1]
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    x_shape, y_shape, regression = list(x_train.shape[1:]), 1, True  # number of features / output dim
    return train_loader, val_loader, test_loader, x_shape, y_shape, regression


# </editor-fold>


# <editor-fold desc="KDDCUP99 DATASET: Detection of network attacks -------------------------------------------------">

def download_extract_kddcup99(path):

    # classes: normal or type of intrusion in 4 main categories
    class_map = {b'normal.': 0,
                 b'back.': 1, b'land.': 2, b'neptune.': 3, b'pod.': 4, b'smurf.': 5, b'teardrop.': 6,  # dos
                 b'buffer_overflow.': 7,  b'loadmodule.': 8, b'perl.': 9, b'rootkit.': 10,  # u2r
                 b'ftp_write.': 11, b'guess_passwd.': 12, b'imap.': 13, b'multihop.': 14,  # r2l
                 b'phf.': 15, b'spy.': 16, b'warezclient.': 17, b'warezmaster.': 18,  # r2l
                 b'ipsweep.': 19, b'nmap.': 20, b'portsweep.': 21, b'satan.': 22  # probing
                 }
    
    # protocol types
    prtcl_map = [b'icmp', b'tcp', b'udp']
    
    # service types
    servc_map = [b'IRC', b'X11', b'Z39_50', b'aol', b'auth', b'bgp', b'courier',
                 b'csnet_ns', b'ctf', b'daytime', b'discard', b'domain',
                 b'domain_u', b'echo', b'eco_i', b'ecr_i', b'efs', b'exec',
                 b'finger', b'ftp', b'ftp_data', b'gopher', b'harvest',
                 b'hostnames', b'http', b'http_2784', b'http_443', b'http_8001',
                 b'imap4', b'iso_tsap', b'klogin', b'kshell', b'ldap', b'link',
                 b'login', b'mtp', b'name', b'netbios_dgm', b'netbios_ns',
                 b'netbios_ssn', b'netstat', b'nnsp', b'nntp', b'ntp_u', b'other',
                 b'pm_dump', b'pop_2', b'pop_3', b'printer', b'private', b'red_i',
                 b'remote_job', b'rje', b'shell', b'smtp', b'sql_net', b'ssh',
                 b'sunrpc', b'supdup', b'systat', b'telnet', b'tftp_u', b'tim_i',
                 b'time', b'urh_i', b'urp_i', b'uucp', b'uucp_path', b'vmnet', b'whois']
    
    # connection states
    cnnst_map = [b'OTH', b'REJ', b'RSTO', b'RSTOS0', b'RSTR', b'S0', b'S1', b'S2', b'S3', b'SF', b'SH']

    # process and save as torch files
    warn('Downloading and preparing KDDCup99 dataset...')

    ds = sklearn.datasets.fetch_kddcup99(data_home=path, percent10=False)
    n_rows, n_cols = ds.data.shape

    x = np.empty(ds.data.shape, dtype=float)
    y = torch.CharTensor(n_rows)

    for i in range(y.shape[0]):
        x[i, 0] = ds.data[i, 0]
        x[i, 4:] = ds.data[i, 4:]
        x[i, 1] = prtcl_map.index(ds.data[i, 1])
        x[i, 2] = servc_map.index(ds.data[i, 2])
        x[i, 3] = cnnst_map.index(ds.data[i, 3])
        y[i] = class_map[ds.target[i]]
    x = torch.from_numpy(x).float()
    x_train, y_train, x_test, y_test = split_dataset(x, y, TEST_RATIO)

    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'KDDCup99.pt'))
    shutil.rmtree(os.path.join(path, 'kddcup99-py3'))  # remove temporary data directory


def load_kddcup99(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, preprocessing='cat2bin, normalize',
                  train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'KDDCup99.pt')):
        download_extract_kddcup99(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'KDDCup99.pt')
    y_train, y_test = y_train.long(), y_test.long()

    if 'cat2bin' in preprocessing:  # change categorical attributes to binary indicators
        num_cats = [3, 70, 11]
        num_cols = sum(num_cats)
        x1_train = torch.zeros(x_train.shape[0], num_cols)
        x1_test = torch.zeros(x_test.shape[0], num_cols)
        for i in range(len(num_cats)):
            for j in range(num_cats[i]):
                col = sum(num_cats[:i]) + j
                x1_train[:, col][x_train[:, i + 1] == j] = 1.
                x1_test[:, col][x_test[:, i + 1] == j] = 1.
        take_cols = [i for i in range(x_train.shape[1]) if i == 0 or i > 3]
        x_train = torch.cat([x_train[:, take_cols], x1_train], dim=1)
        x_test = torch.cat([x_test[:, take_cols], x1_test], dim=1)

    if 'normalize' in preprocessing:  # zero mean 1 std
        if 'cat2bin' in preprocessing:  # only first 38 features are quantitative, remaining are cat indicators
            normalize_data(x_train[:, :38], x_test[:, :38])
        else:
            normalize_data(x_train, x_test)
    elif 'squash' in preprocessing:  # squash data range to [-1 1]
        squash_data(x_train, x_test)

    # default objective is identify attack method: no attack or one of 21 attack methods
    # this can be changed to 2 classes or 5 classes objective as follows
    y_shape = 23
    if 'class:yesno' in preprocessing:  # objective is prediction if it is network attack or not
        y_train[y_train > 0] = 1
        y_test[y_test > 0] = 1
        y_shape = 2
    elif 'class:groups' in preprocessing:  # objective is identify the group of attack method
        y_train[1 <= y_train <= 6] = 1  # dos
        y_train[7 <= y_train <= 10] = 2  # u2r
        y_train[11 <= y_train <= 18] = 3  # r2l
        y_train[y_train >= 19] = 4  # probing

        y_test[1 <= y_test <= 6] = 1  # dos
        y_test[7 <= y_test <= 10] = 2  # u2r
        y_test[11 <= y_test <= 18] = 3  # r2l
        y_test[y_test >= 19] = 4  # probing
        y_shape = 5

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    x_shape, regression = list(x_train.shape[1:]), False  # number of features / output dim
    return train_loader, val_loader, test_loader, x_shape, y_shape, regression


# </editor-fold>


# <editor-fold desc="HOUSING DATASET: Prediction of median house value for California districts ----------------------">


def download_extract_housing(path):
    warn('Downloading and preparing California Housing dataset...')

    ds = sklearn.datasets.fetch_california_housing(data_home=path)

    x = torch.from_numpy(ds.data).float()
    y = torch.from_numpy(ds.target).float()

    x_train, y_train, x_test, y_test = split_dataset(x, y, TEST_RATIO)

    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'Housing.pt'))
    os.unlink(os.path.join(path, 'cal_housing_py3.pkz'))


def load_housing(cuda=False, num_workers=1, shuffle=True, validation_ratio=0, preprocessing='normalize',
                 train_batch_size=256, test_batch_size=256, path=config['DEFAULT']['datasets']):
    if not os.path.exists(os.path.join(path, 'Housing.pt')):
        download_extract_housing(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'Housing.pt')
    y_train, y_test = y_train.long(), y_test.long()

    if 'normalize' in preprocessing:  # zero mean 1 std
        normalize_data(x_train, x_test)
    elif 'squash' in preprocessing:  # squash data range to [-1 1]
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    x_shape, y_shape, regression = list(x_train.shape[1:]), 1, True  # number of features / output dim
    return train_loader, val_loader, test_loader, x_shape, y_shape, regression


# </editor-fold>


# <editor-fold desc="sklearn toy datasets ----------------------------------------------------------------------------">

def load_toy_dataset(name, cuda=False, num_workers=1, shuffle=True, validation_ratio=0,
                     preprocessing='normalize', train_batch_size=16, test_batch_size=16):
    """toy datasets: name = boston | iris | diabetes | digits | wine | breast_cancer
    boston: x_dim=13, y_range=[5. - 50.].  diabetes: x_dim=10, y_range=[25. - 346.]
    other datasets are classification, x_dim: iris=4, digits=64, wine=13, breast_cancer=30
    n_samples: boston=506, iris=150, diabetes=442, digits=1797, wine=178, breast_cancer=569"""
    ds_params = {'boston': {'regression': True, 'num_out': 1},
                 'iris': {'regression': False, 'num_out': 3},
                 'diabetes': {'regression': True, 'num_out': 1},
                 'digits': {'regression': False, 'num_out': 10},
                 'wine': {'regression': False, 'num_out': 3},
                 'breast_cancer': {'regression': False, 'num_out': 2},
                 }
    assert name in ds_params.keys(), 'Unknown dataset ({})!'.format(name)

    load_func = getattr(sklearn.datasets, 'load_' + name)
    ds = load_func()

    x = torch.from_numpy(ds.data).float()
    if ds_params[name]['regression']:
        y = torch.from_numpy(ds.target).float()
    else:
        y = torch.from_numpy(ds.target).long()

    x_train, y_train, x_test, y_test = split_dataset(x, y, TEST_RATIO)  # 20% for test set

    if 'normalize' in preprocessing:  # zero mean 1 std
        normalize_data(x_train, x_test)
    elif 'squash' in preprocessing:  # squash data range to [-1 1]
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    x_shape = list(x_train.shape[1:])
    num_out, regression = ds_params[name]['num_out'], ds_params[name]['regression']
    return train_loader, val_loader, test_loader, x_shape, num_out, regression


# </editor-fold>


# <editor-fold desc="MiniBooNE DATASET: MiniBooNE particle identification---------------------------------------------">


def download_extract_MiniBooNE(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00199/"
    filename = "MiniBooNE_PID.txt"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 90.:
        # download if file not exists or smaller than 90MB -> appears broken
        file_path = download_file(url, filename, path)

    # process and save as torch files
    warn('Preparing MiniBooNE dataset...')

    xs = []
    ys = []
    # file_path = file_path.replace('.zip', '')
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            if i == 0:  # first line contains number of signal events and bgr events
                r = line.split(' ')
                n_sig, n_bgr = int(r[1]), int(r[2])
            else:
                r = line.strip().replace('  ', ' ').split(' ')
                xs.append([float(x) for x in r])
                ys.append(1 if i <= n_sig else 0)
            i += 1

    assert i - 1 == n_sig + n_bgr, 'Expected {} data rows, got {} rows'.format(n_sig + n_bgr, i - 1)

    x = torch.from_numpy(np.array(xs, float)).float()
    y = torch.from_numpy(np.array(ys, int)).int()

    x_train, y_train, x_test, y_test = split_dataset(x, y, TEST_RATIO)

    os.unlink(file_path)
    file_path = file_path.replace('_PID.txt', '.pt')
    torch.save((x_train, y_train, x_test, y_test), file_path)


def load_miniboone(cuda=False, num_workers=1, shuffle=True, validation_ratio=0,
                   preprocessing='normalize', train_batch_size=256, test_batch_size=256,
                   path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'MiniBooNE.pt')):
        download_extract_MiniBooNE(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'MiniBooNE.pt')
    y_train, y_test = y_train.long(), y_test.long()
    x_shape = list(x_train.shape[1:])

    if 'regression' in preprocessing:
        y_shape = 1
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1
    else:  # classification, change y to {0, 1}
        y_shape = 2

    if 'normalize' in preprocessing:
        normalize_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_shape, y_shape, 'regression' in preprocessing

# </editor-fold>


# <editor-fold desc="TRANSCODING DATASET: Prediction of video transcoding time based on params -----------------------">

def load_BlogFeedback_csv(file):
    xs, ys = [], []
    with open(file, 'r') as f:
        for line in f:
            r = line.strip().replace(' ', '').split(',')
            assert len(r) == 281, 'Expected 281 features, got {}! @line {} @file {}'.format(len(r), len(ys), file)
            xs.append([float(x) for x in r[:-1]])
            ys.append(float(r[-1]))
    x = torch.from_numpy(np.array(xs, float)).float()
    y = torch.from_numpy(np.array(ys, int)).int()
    os.unlink(file)
    return x, y


def download_extract_BlogFeedback(path):

    url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00304/"
    filename = "BlogFeedback.zip"
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path) or os.path.getsize(file_path) / 1e6 < 2.4:
        # download if file not exists or smaller than 2MB -> appears broken
        file_path = download_file(url, filename, path)

    with zipfile.ZipFile(file_path) as myzip:
        myzip.extractall(path)

    # process and save as torch files
    warn('Preparing BlogFeedback dataset...')

    x_train, y_train = load_BlogFeedback_csv(os.path.join(path, 'blogData_train.csv'))

    xs, ys = [], []
    for file in os.listdir(path):
        if 'blogData_test-2012.' in file:
            x, y = load_BlogFeedback_csv(os.path.join(path, file))
            xs.append(x)
            ys.append(y)
    x_test = torch.cat(xs)
    y_test = torch.cat(ys)

    torch.save((x_train, y_train, x_test, y_test), os.path.join(path, 'BlogFeedback.pt'))


def load_blogfeedback(cuda=False, num_workers=1, shuffle=True, validation_ratio=0,
                      preprocessing='normalize', train_batch_size=256, test_batch_size=256,
                      path=config['DEFAULT']['datasets']):

    if not os.path.exists(os.path.join(path, 'BlogFeedback.pt')):
        download_extract_BlogFeedback(path)

    x_train, y_train, x_test, y_test = torch.load(path + 'BlogFeedback.pt')
    y_train = y_train.long()
    y_test = y_test.long()

    x_shape, y_shape, regression = list(x_train.shape[1:]), 1, True  # number of features / output dim

    if 'y-mean' in preprocessing or 'y-middle' in preprocessing:
        # center around mean=7 or middle of the range=712, range is [0 - 1424]
        y_mean = 7 if 'y-mean' in preprocessing else 712
        y_train -= y_mean
        y_test -= y_mean

    if 'normalize' in preprocessing:
        normalize_data(x_train, x_test)
    elif 'squash' in preprocessing:
        squash_data(x_train, x_test)

    if validation_ratio > 0:
        x_train, y_train, x_val, y_val = split_dataset(x_train, y_train, validation_ratio)
    else:
        x_val, y_val = x_test.clone(), y_test.clone()

    train_loader = DataLoader(x_train, y_train, cuda, num_workers, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(x_val, y_val, cuda, num_workers, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, cuda, num_workers, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_shape, y_shape, regression

# </editor-fold>


# ==================== MAIN =================================================================== #

if __name__ == '__main__':
    print(config['DEFAULT']['datasets'])
    # dl = DataLoader(np.arange(12).reshape(3, 2, 2), np.array([0, 1, 2]), False, 2)
    # dl2 = dl.new(batch_size=10)
    # l1, l2, l3, shape_x, shape_y, regrsion = load_blogfeedback(validation_ratio=0.2)
    # print(shape_x, shape_y)
    # download_extract_BlogFeedback(config['DEFAULT']['datasets'])
