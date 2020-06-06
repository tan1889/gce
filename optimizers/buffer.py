import torch.nn as nn
import torch
import math
import random


def get_random_batch_ids(n_samples, batch_size, min_last_batch_size=2):
    """generate random batches of indices. indices must be non duplicate integers smaller than n_samples."""
    ids = torch.randperm(n_samples)
    batch_ids = []
    i = 0
    while i < n_samples:
        i1 = min(i + batch_size, n_samples)
        if i1 < n_samples or i1 - i >= min_last_batch_size:
            batch_ids.append(ids[i:i1])
        i = i1
    return batch_ids


class DataLoaderBuffer:
    r"""
    Caching the input data and output of sum_net and transform_net can
    provide ~ 10x speed up for training, so we should cache when it's possible.

    Arguments:
        data_loader: data loader for the dataset
        caching: 1=enabled, 0=disabled (if not enough memory or randomly transformed input -> cant cache),
        device: indicator of where the data should be stored: cpu or cuda
        sum_net: for wgn algorithm, this is the convex combination of modules that has been trained,
            the output of this net will be combined with the output of the module to be trained.
        transform_net: this can be either a features extractor or the trained modules in the case of dgnet.
            The output of this net is used to train the module that is to be trained.

    ..notes:    sum_net, transform_net must be already on the same device indicated by arg device.
                If transform_net is features_net like mnist.pt, we can have significant boost in speed if we
                cache it at cpu data_utils level
    """

    def __init__(self, data_loader, caching, device, sum_net=None, transform_net=None):
        assert data_loader.cacheable or caching <= 0, 'Caching is enabled but dataset can not be cached!'
        self.device = device
        self.caching = caching
        self.n_samples = len(data_loader.dataset)
        self.batch_size = data_loader.batch_size
        self.cache = None
        self.index = 0
        self.batch_ids = None

        if caching <= 0:
            self.data_loader = data_loader
            self.sum_net = sum_net
            self.transform_net = transform_net
        else:
            if sum_net is not None: sum_net.eval()  # change to eval mode
            if transform_net is not None: transform_net.eval()
            xs, ys, ss = [], [], []
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                if transform_net is not None:
                    x = transform_net(x)
                if sum_net is not None and len(sum_net) > 0:
                    s = sum_net(x)
                    ss.append(s)
                xs.append(x)
                ys.append(y)

            xs = torch.cat(xs)
            ys = torch.cat(ys)
            if ss: ss = torch.cat(ss)
            self.cache = {'xs': xs, 'ys': ys, 'ss': ss}

    def __iter__(self):
        if self.cache is None:
            self.data_iterator = iter(self.data_loader)
        else:
            self.index = 0
            if self.caching <= 2 or self.batch_ids is None:  # reshuffle ids in batch_ids
                self.batch_ids = get_random_batch_ids(self.n_samples, self.batch_size)
            elif self.caching == 3:  # reshuffle the order of batches only
                random.shuffle(self.batch_ids)
        return self

    def __next__(self):
        if self.cache:
            if self.index >= len(self.batch_ids):
                raise StopIteration()
            else:
                ids = self.batch_ids[self.index]
                x = self.cache['xs'][ids]
                y = self.cache['ys'][ids]
                s = None
                if self.cache['ss']:
                    s = self.cache['ss'][ids]
                self.index += 1
        else:
            try:
                (x, y) = next(self.data_iterator)
            except StopIteration:
                raise StopIteration()
            x, y = x.to(self.device), y.to(self.device)
            if self.transform_net is not None:
                x = self.transform_net(x)
            s = None
            if self.sum_net is not None and len(self.sum_net) > 0:
                s = self.sum_net(x)
        return x, y, s

    def dispose(self):
        if self.cache:
            del self.cache[:]


def class2vec(y, k):
    """Convert from a vector of classes 'y' to the corresponding vector of vector encoding of classes.
    k is the number of classes. Returns yk with dimension len(y) x k, where yk[i] the vector encode of class y[i]
    """
    n = y.size(0)
    if k <= 1:  # single output = regression
        yk = y.view(n, 1).float()
    else:  # multi output -> classification
        yk = y.new(n, k).float().fill_(0.0)
        yk[range(n), y] = 1.0
    return yk


class DataLoaderBufferFW:
    r"""
    Similar as DataLoaderBuffer above, but specifically for Frank Wolfe algorithms. So, apart from x, y,
    z=sum_net(x), we also cache loss and grad of criterion(z, y). Note: y is cached in the form appropriate to
    calculate loss. For classification task using MSE, this is memory inefficient: y converted by class2vec,
    if there are many classes -> inefficient. This is however uncommon (experimental) situation as we often use
    MSE for classification, so for convenience we leave it like this at the moment.

    Arguments: same as above, and
        criterion: loss function
        output_dim: number of classes for classification or 1 for regression
        subsample: ratio of the training data to be used to train. Default is 1. XGBoost claims using subsample < 1
            have the regularization effect. We tested on toy datasets from sklearn and subsample < 1 only worsen
            the solution for all cases. Thus, this option is left here for experimental purpose.
        requires_grad: for calculating losses on validation/test set, grad is not required, set this to False to save
            some memory and computation.

    Note: Even if data is transformed/deskewed, we can still use cache with FW as it's only for each module. So,
    no_cache here should be used only when there is insufficient memory. In practice, if we cache each module with a
    different version of the input (by a transformation), then the final result is not good. So, for transformation,
    it is better to first expand the data, e.g. from 60K images to 300K images, then cache the whole thing.
    """

    def __init__(self, data_loader, caching, device, criterion, output_dim,
                 transform_net=None, subsample=1., requires_grad=True):
        assert data_loader.cacheable or caching <= 0, 'Caching is enabled but dataset can not be cached!'
        assert 0.1 <= subsample <= 1.0, 'subsample must be in [0.1, 1.0]!'
        assert subsample == 1. or caching > 0, 'subsample < 1 can only work with caching > 0!'
        self.device = device
        self.criterion = criterion
        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')
        self.out_dim = output_dim
        self.requires_grad = requires_grad
        self.batch_size = data_loader.batch_size
        n_data = len(data_loader.dataset)
        self.n_samples = math.ceil(n_data * subsample)
        self.caching = caching
        self.data = self.cache = None
        self.index = 0
        self.batch_ids = None
        self.sum_net = None

        if transform_net is not None:
            transform_net.eval()

        if caching <= 0:
            self.data_loader = data_loader
            self.transform_net = transform_net
        else:  # load all data to memory (cuda or cpu appropriately)
            xs, ys = [], []
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                if transform_net is not None:
                    x = transform_net(x)
                if type(criterion) is not nn.CrossEntropyLoss:
                    y = class2vec(y, output_dim)  # form of y depends on loss type
                xs.append(x)
                ys.append(y)
            xs = torch.cat(xs)
            ys = torch.cat(ys)
            take_ids = torch.randperm(n_data)[:self.n_samples]  # indices of (n_samples) sampled rows
            xs = xs[take_ids]
            ys = ys[take_ids]
            self.data = {'xs': xs, 'ys': ys}

    def update(self, sum_net):
        self.batch_ids = get_random_batch_ids(self.n_samples, self.batch_size)
        self.sum_net = sum_net
        if self.data and self.caching > 1:  # data cached, cache batches only if caching > 1
            if self.cache: self.cache.clear()
            zs, zs_grad, losses = [], [], []
            # cache values for each data point so it can be used in random batches
            i = 0
            while i < self.n_samples:
                i1 = min(i + self.batch_size, self.n_samples)
                x = self.data['xs'][i:i1]
                y = self.data['ys'][i:i1]
                z = sum_net(x)
                z.requires_grad = self.requires_grad
                # loss = 0.5 * self.l1(z, y) + 0.5 * self.l2(z, y)  # 25/7 temporary to test mixed loss for pfw
                loss = self.criterion(z, y)
                if self.requires_grad:
                    loss.mean().backward()
                    z_grad = x.size(0) * z.grad.data / self.n_samples  # rescale grad to reflect the whole training set
                    zs_grad.append(z_grad)
                zs.append(z.data)  # we only need to cache z values
                losses.append(loss.data)
                i = i1
            zs = torch.cat(zs)
            losses = torch.cat(losses)
            if self.requires_grad:
                zs_grad = torch.cat(zs_grad)
            self.cache = {'zs': zs, 'zs_grad': zs_grad, 'losses': losses}

    def eval_loss(self, alpha=0, d=None):
        """given direction d and coefficient alpha, calculate loss of function f(X) + alpha*d(X). X=training_samples.
        If d or alpha are not specified -> eval loss of function f(X) (i.e. alpha=d=0)"""
        assert d or alpha == 0, 'both d and alpha must be specified to calculate loss with a direction'
        assert self.sum_net, 'function sum_net is not yet loaded'

        loss = 0.
        if self.data:
            for ids in self.batch_ids:
                x = self.data['xs'][ids]
                y = self.data['ys'][ids]
                z = self.cache['zs'][ids] if self.cache else self.sum_net(x)
                if alpha != 0:
                    z += alpha * d(x, z)
                loss += self.criterion(z, y).sum().item()
            loss /= self.n_samples
        else:  # no cache -> calculate directly from data loader
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.transform_net is not None:
                    x = self.transform_net(x)
                z = self.sum_net(x)
                if alpha != 0:
                    z += alpha * d(x, z)
                if type(self.criterion) is not nn.CrossEntropyLoss:
                    y = class2vec(y, z.size(1))  # form of y depends on loss type
                loss += self.criterion(z, y).sum().item()
            loss /= self.n_samples

        return loss

    def __iter__(self):
        assert self.sum_net or self.cache, 'Must assign sum_net via .update() first!'
        if self.data is None:  # cache and data are not in memory -> get directly from data_loader
            self.data_iterator = iter(self.data_loader)
        else:
            self.index = 0
            if not self.cache or self.caching < 3:
                self.batch_ids = get_random_batch_ids(self.n_samples, self.batch_size)
            elif self.caching == 3:
                random.shuffle(self.batch_ids)
        return self

    def __next__(self):
        if self.cache:
            if self.index >= len(self.batch_ids):
                raise StopIteration()
            else:
                ids = self.batch_ids[self.index]
                x = self.data['xs'][ids]
                y = self.data['ys'][ids]
                loss = torch.mean(self.cache['losses'][ids]).item()
                z = self.cache['zs'][ids]
                z_grad = None
                if self.requires_grad:
                    z_grad = self.cache['zs_grad'][ids]
                self.index += 1
        elif self.data:  # data x, y are cached
            if self.index >= len(self.batch_ids):
                raise StopIteration()
            else:
                ids = self.batch_ids[self.index]
                x = self.data['xs'][ids]
                y = self.data['ys'][ids]
                z = self.sum_net(x)
                z.requires_grad = self.requires_grad
                loss = self.criterion(z, y).mean()
                z_grad = None
                if self.requires_grad:
                    loss.backward()
                    z_grad = x.size(0) * z.grad.data / self.n_samples  # rescale grad to reflect the whole training set
                z = z.data
                loss = loss.item()
                self.index += 1
        else:
            try:
                (x, y) = next(self.data_iterator)
            except StopIteration:
                raise StopIteration()
            x, y = x.to(self.device), y.to(self.device)
            if self.transform_net is not None:
                x = self.transform_net(x)
            z = self.sum_net(x)  # output of sum_net
            if type(self.criterion) is not nn.CrossEntropyLoss:
                y = class2vec(y, z.size(1))  # form of y depends on loss type
            z.requires_grad = self.requires_grad
            loss = self.criterion(z, y).mean()
            z_grad = None
            if self.requires_grad:
                loss.backward()
                z_grad = x.size(0) * z.grad.data / self.n_samples  # rescale grad to reflect the whole training set
            z = z.data  # we only need to cache z values
            loss = loss.item()
        return x, y, z, z_grad, loss


class DataLoaderBufferGRN:
    r"""
    Similar as DataLoaderBufferFW above, but specifically for Greedy ResNet. So, we cache all gradients of g_k
    of each residual module of the Greedy ResNet.
    iter over batches depends on the mode: .train(g_index=5) will output input, grad, output for residual module 5
    (zero based indexing) .eval() sets self.g_index=-1 and return output and loss of the function
    """

    def __init__(self, data_loader, caching, device, criterion, output_dim,
                 transform_net=None, subsample=1., requires_grad=True):
        assert data_loader.cacheable or caching <= 0, 'Caching is enabled but dataset can not be cached!'
        assert 0.1 <= subsample <= 1.0, 'subsample must be in [0.1, 1.0]!'
        assert subsample == 1. or caching > 0, 'subsample < 1 can only work with caching > 0!'
        self.device = device
        self.criterion = criterion
        self.out_dim = output_dim
        self.requires_grad = requires_grad
        self.batch_size = data_loader.batch_size
        n_data = len(data_loader.dataset)
        self.n_samples = math.ceil(n_data * subsample)
        self.caching = caching
        self.data = self.cache = None
        self.index = 0
        self.batch_ids = None
        self.gresnet = None
        self.g_index = -1  # select which residual module (and corresponding in/output to be selected for getting data

        if transform_net is not None:
            transform_net.eval()

        if caching <= 0:
            self.data_loader = data_loader
            self.transform_net = transform_net
        else:  # load all data to memory (cuda or cpu appropriately)
            xs, ys = [], []
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                if transform_net is not None:
                    x = transform_net(x)
                if type(criterion) is not nn.CrossEntropyLoss:
                    y = class2vec(y, output_dim)  # form of y depends on loss type
                xs.append(x)
                ys.append(y)
            xs = torch.cat(xs)
            ys = torch.cat(ys)
            take_ids = torch.randperm(n_data)[:self.n_samples]  # indices of (n_samples) sampled rows
            xs = xs[take_ids]
            ys = ys[take_ids]
            self.data = {'xs': xs, 'ys': ys}

    def update(self, gresnet):
        self.batch_ids = get_random_batch_ids(self.n_samples, self.batch_size)
        self.gresnet = gresnet

        if self.data and self.caching > 1:  # data cached, cache batches only if caching > 1
            if self.cache: self.cache.clear()
            zs, losses = [], []
            # grads[k] is the grad of g_k over each data point, so it is a tensor of N x g_k.dim_out
            grads = [[] for _ in range(len(gresnet))]
            fs = [[] for _ in range(len(gresnet))]
            # cache values for each data point so it can be used in random batches
            i = 0
            while i < self.n_samples:
                i1 = min(i + self.batch_size, self.n_samples)
                x = self.data['xs'][i:i1]
                y = self.data['ys'][i:i1]
                g = []
                for k in range(len(gresnet)):
                    g_k = gresnet[k](x)
                    g_k.requires_grad = self.requires_grad
                    g.append(g_k)
                    x = g_k + x[:, :gresnet[k].dim_out]
                    fs[k].append(x.data)

                loss = self.criterion(x, y)
                if self.requires_grad:
                    loss.mean().backward()
                    # add grad of g_k to grads
                    for k in range(len(gresnet)):
                        # rescale grad to reflect the whole training set
                        grads[k].append(x.size(0) * g[k].grad.data / self.n_samples)
                losses.append(loss.data)
                i = i1
            losses = torch.cat(losses)
            for k in range(len(gresnet)):
                fs[k] = torch.cat(fs[k])
                if self.requires_grad:
                    grads[k] = torch.cat(grads[k])
            self.cache = {'fs': fs, 'grads': grads, 'losses': losses}

    def eval_loss(self, alpha=0, d=None):
        """given direction d and coefficient alpha, calculate loss of function f(X) + alpha*d(X), X=training_samples.
        If direction and alpha are not specified -> eval loss of function f(X) (i.e. alpha = direction = 0"""
        assert d or alpha == 0, 'both direction and alpha must be specified to calculate loss with a direction'
        assert self.gresnet, 'function gresnet is not yet loaded'

        loss = 0.
        if self.data:
            for ids in self.batch_ids:
                x = self.data['xs'][ids]
                y = self.data['ys'][ids]
                if alpha != 0:
                    r = self.gresnet[0](x)  # output of residual module
                    fw_mdl, aw_mdl, _, _ = d[0]
                    z = (1. - alpha) * r + alpha * (fw_mdl(x)) + x[:, :self.gresnet[0].dim_out]
                    for i in range(1, len(self.gresnet)):
                        r = self.gresnet[i](z)
                        fw_mdl, aw_mdl, _, _ = d[i]
                        z = (1. - alpha) * r + alpha * (fw_mdl(z)) + z[:, :self.gresnet[i].dim_out]
                else:
                    z = self.gresnet(x)
                loss += self.criterion(z, y).sum().item()
            loss /= self.n_samples
        else:  # no cache -> calculate directly from data loader
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.transform_net is not None:
                    x = self.transform_net(x)
                if alpha != 0:
                    r = self.gresnet[0](x)  # output of residual module
                    z = r + alpha * d[0](x, r) + x[:, :self.gresnet[0].dim_out]
                    for i in range(1, len(self.gresnet)):
                        r = self.gresnet[i](z)
                        z = r + alpha * d[i](z, r) + z[:, :self.gresnet[i].dim_out]
                else:
                    z = self.gresnet(x)
                if type(self.criterion) is not nn.CrossEntropyLoss:
                    y = class2vec(y, z.size(1))  # form of y depends on loss type
                loss += self.criterion(z, y).sum().item()
            loss /= self.n_samples

        return loss

    def eval(self):
        """set eval mode"""
        self.g_index = -1

    def train(self, g_index):
        """set training mode for the specific resnet module g_index"""
        assert 0 <= g_index <= len(self.gresnet)
        self.g_index = g_index

    def __iter__(self):
        assert self.gresnet or self.cache, 'Must assign sum_net via .update() first!'
        if self.data is None:  # cache and data are not in memory -> get directly from data_loader
            self.data_iterator = iter(self.data_loader)
        else:
            self.index = 0
            if not self.cache or self.caching < 3:
                self.batch_ids = get_random_batch_ids(self.n_samples, self.batch_size)
            elif self.caching == 3:
                random.shuffle(self.batch_ids)
        return self

    def __next__(self):
        if self.cache:
            if self.index >= len(self.batch_ids):
                raise StopIteration()
            else:
                ids = self.batch_ids[self.index]
                y = self.data['ys'][ids]
                z = self.cache['fs'][self.g_index][ids]
                if self.g_index < 0:  # eval mode, only need target y, output z, and loss
                    x = z_grad = None
                    loss = torch.mean(self.cache['losses'][ids]).item()
                else:  # train mode, need x (input of layer g_index, and grad
                    loss = z_grad = None
                    x = self.data['xs'][ids] if self.g_index == 0 else self.cache['fs'][self.g_index - 1][ids]
                    if self.requires_grad:
                        z_grad = self.cache['grads'][self.g_index][ids]
                self.index += 1
        elif self.data:  # data x, y are cached
            if self.index >= len(self.batch_ids):
                raise StopIteration()
            else:
                ids = self.batch_ids[self.index]
                x = self.data['xs'][ids]
                y = self.data['ys'][ids]
                g = []
                for k in range(len(self.gresnet)):  # gresnet consists of k residual module g_0 ... g_k-1
                    g_k = self.gresnet[k](x)
                    g_k.requires_grad = self.requires_grad
                    x = g_k + x[:, :self.gresnet[k].dim_out]
                    g.append(g_k)
                loss = self.criterion(x, y).mean()
                z_grad = None
                if self.requires_grad:
                    loss.backward()
                    # add grad of g_k to grads
                    z_grad = x.size(0) * g[self.g_index].grad.data / self.n_samples
                z = x.data  # we only need to cache z values
                loss = loss.item()
                self.index += 1
        else:
            try:
                (x, y) = next(self.data_iterator)
            except StopIteration:
                raise StopIteration()
            x, y = x.to(self.device), y.to(self.device)
            if self.transform_net is not None:
                x = self.transform_net(x)
            g = []
            for k in range(len(self.gresnet)):  # gresnet consists of k residual module g_0 ... g_k-1
                g_k = self.gresnet[k](x)
                g_k.requires_grad = self.requires_grad
                x = g_k + x[:, :self.gresnet[k].dim_out]
                g.append(g_k)
            if type(self.criterion) is not nn.CrossEntropyLoss:
                y = class2vec(y, x.size(1))  # form of y depends on loss type
            loss = self.criterion(x, y).mean()
            z_grad = None
            if self.requires_grad:
                loss.backward()
                # add grad of g_k to grads
                z_grad = x.size(0) * g[self.g_index].grad.data / self.n_samples
            z = x.data  # we only need to cache z values
            loss = loss.item()
            self.index += 1
        return x, y, z, z_grad, loss
