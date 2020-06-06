"""
Implementation of some common neural networks to be trained classically by SGD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizers.sgd import TorchSGD as SGD


class NN1(nn.Module):
    """Implements a 1 linear layer neural net (equivalent to linear regressor / classifier"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN1, self).__init__()
        args.module_size = 1
        self.f = afunc
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class ConvexNeuron(nn.Module):
    """Implements  alpha^T tanh(u^T x), sum(alpha)=B alpha > 0.
    This is like the Neuron module in models.defs, but here trained simultaneously to compare the two approaches"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(ConvexNeuron, self).__init__()
        assert args.module_size >= 1, 'module_size must be at least 1'
        assert args.activation == 'tanh', 'activation must be tanh for ConvexNeuron module'
        self.bound = args.bound
        # init network weights
        stdv = 1. / (dim_in ** 0.5)
        self.weight = nn.Parameter(torch.Tensor(args.module_size, dim_in, dim_out))
        self.bias = nn.Parameter(torch.Tensor(args.module_size, 1, dim_out))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # init convex weights
        self.alpha = nn.Parameter(0.2 * torch.ones(args.module_size, 1, 1))

    def forward(self, x):
        a = self.alpha.abs()  # normalize using B/K + |a_i| / (K + sum(a))
        cw = (self.bound / a.size(0) + a) / (1. + a.sum())  # cw_i > 0, sum(cw) = B
        x = x.view(x.size(0), -1)
        xs = [x.mm(self.weight[i]) for i in range(self.weight.size(0))]
        xs = torch.stack(xs, dim=0)
        x = F.tanh(xs + self.bias) * cw
        x = x.sum(dim=0)
        return x


class NN2(nn.Module):
    """Implements a standard 1 hidden layers neural net"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN2, self).__init__()
        assert args.module_size >= 1, 'module_size must be at least 1'
        self.f = afunc
        self.fc1 = nn.Linear(dim_in, args.module_size)
        self.fc2 = nn.Linear(args.module_size, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc2(self.f(self.fc1(x)))
        return out


class NN2h(nn.Module):
    """Same as NN2 but use HardTanh to bound the output range"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN2h, self).__init__()
        assert args.module_size >= 1, 'module_size must be at least 1'
        self.f = afunc
        self.bound = args.bound
        self.fc1 = nn.Linear(dim_in, args.module_size)
        self.fc2 = nn.Linear(args.module_size, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc2(self.f(self.fc1(x)))
        out = F.hardtanh(x, min_val=-self.bound, max_val=self.bound)
        return out


class CN2h(nn.Module):
    """Implements a convex combination of NN2h modules: alpha^T x NN2Hs(x), sum(alpha) <= B alpha > 0.
    args.modules defines the number of module, args.module_size define number of hidden units for each NN2H module."""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(CN2h, self).__init__()
        assert args.module_size >= 1, 'module_size must be at least 1'
        self.f = afunc
        self.bound = args.bound
        self.n_modules = args.n_modules
        self.module_size = args.module_size
        # init network weights
        self.W1 = nn.Parameter(torch.Tensor(self.n_modules, dim_in, self.module_size))
        self.b1 = nn.Parameter(torch.Tensor(self.n_modules, 1, self.module_size))
        self.W2 = nn.Parameter(torch.Tensor(self.n_modules, self.module_size, dim_out))
        self.b2 = nn.Parameter(torch.Tensor(self.n_modules, 1, dim_out))
        stdv = 1. / (dim_in ** 0.5)
        self.W1.data.uniform_(-stdv, stdv)
        self.b1.data.uniform_(-stdv, stdv)
        stdv = 1. / (self.module_size ** 0.5)
        self.W2.data.uniform_(-stdv, stdv)
        self.b2.data.uniform_(-stdv, stdv)
        # init convex weights
        self.alpha = nn.Parameter(0.2 * torch.ones(self.n_modules, 1, 1))

    def forward(self, x):
        a = self.alpha.abs()  # normalize using B/K + |a_i| / (K + sum(a))
        cw = (self.bound / a.size(0) + a) / (1. + a.sum())  # cw_i > 0, sum(cw) <= B
        x = x.view(x.size(0), -1)
        x = self.f(x.matmul(self.W1) + self.b1)
        x = x.matmul(self.W2) + self.b2
        # x = F.hardtanh(x, min_val=-self.bound, max_val=self.bound)
        x = F.hardtanh(x, min_val=-self.bound, max_val=self.bound)
        x = x * cw
        x = x.sum(dim=0)
        return x


class CN2L2(nn.Module):
    """Implements a convex combination of NN2 modules: alpha^T x NN2s(x), sum(alpha)=1 alpha > 0.
    bound is implicit by L2 regularization
    args.modules defines the number of module, args.module_size define number of hidden units for each NN2H module."""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(CN2L2, self).__init__()
        assert args.module_size >= 1, 'module_size must be at least 1'
        self.f = afunc
        self.bound = args.bound
        self.n_modules = args.n_modules
        self.module_size = args.module_size
        # init network weights
        self.W1 = nn.Parameter(torch.Tensor(self.n_modules, dim_in, self.module_size))
        self.b1 = nn.Parameter(torch.Tensor(self.n_modules, 1, self.module_size))
        self.W2 = nn.Parameter(torch.Tensor(self.n_modules, self.module_size, dim_out))
        self.b2 = nn.Parameter(torch.Tensor(self.n_modules, 1, dim_out))
        stdv = 1. / (dim_in ** 0.5)
        self.W1.data.uniform_(-stdv, stdv)
        self.b1.data.uniform_(-stdv, stdv)
        stdv = 1. / (self.module_size ** 0.5)
        self.W2.data.uniform_(-stdv, stdv)
        self.b2.data.uniform_(-stdv, stdv)
        # init convex weights
        self.alpha = nn.Parameter(0.2 * torch.ones(self.n_modules, 1, 1))

    def forward(self, x):
        a = self.alpha.abs()  # normalize using B/K + |a_i| / (K + sum(a))
        cw = (1. + a) / (1. + a.sum())  # cw_i > 0, sum(cw) <= B
        x = x.view(x.size(0), -1)
        x = self.f(x.matmul(self.W1) + self.b1)
        x = x.matmul(self.W2) + self.b2
        x = x * cw * self.bound / self.n_modules
        x = x.sum(dim=0)
        return x


class LN2(nn.Module):  # linear combination of NN2
    """Implements a convex combination of NN2h modules: alpha^T x NN2Hs(x), sum(alpha) <= B alpha > 0.
    args.modules defines the number of module, args.module_size define number of hidden units for each NN2H module."""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(LN2, self).__init__()
        assert args.module_size >= 1, 'module_size must be at least 1'
        self.f = afunc
        self.n_modules = args.n_modules
        self.module_size = args.module_size
        # init network weights
        self.W1 = nn.Parameter(torch.Tensor(self.n_modules, dim_in, self.module_size))
        self.b1 = nn.Parameter(torch.Tensor(self.n_modules, 1, self.module_size))
        self.W2 = nn.Parameter(torch.Tensor(self.n_modules, self.module_size, dim_out))
        self.b2 = nn.Parameter(torch.Tensor(self.n_modules, 1, dim_out))
        # self.a = nn.Parameter(torch.Tensor(self.n_modules, 1, 1))  # change name to self.alpha will be regularized
        self.bound = args.bound
        stdv = 1. / (dim_in ** 0.5)
        # self.a.data.uniform_(-stdv, stdv)
        self.W1.data.uniform_(-stdv, stdv)
        self.b1.data.uniform_(-stdv, stdv)
        stdv = 1. / (self.module_size ** 0.5)
        self.W2.data.uniform_(-stdv, stdv)
        self.b2.data.uniform_(-stdv, stdv)
        # init linear combination weights

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.f(x.matmul(self.W1) + self.b1)
        x = x.matmul(self.W2) + self.b2
        # x = x * self.a
        x = x * self.bound / self.n_modules  # scale to match the scale of output y_max
        x = x.sum(dim=0)
        return x


class NN3(nn.Module):
    """Implements a standard 2 hidden layers neural net"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN3, self).__init__()
        assert args.module_size >= 3, 'module_size must be at least 3'
        self.f = afunc
        nh = args.module_size // 2
        self.fc1 = nn.Linear(dim_in, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))
        return out


class NN3t(nn.Module):
    """Standard 2 hidden layers neural net with B*tanh applied to output layer to bound its range.
    This is not as good as using hardtanh to bound the output like in NN3h"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN3t, self).__init__()
        assert args.module_size >= 3, 'module_size must be at least 3'
        self.f = afunc
        self.bound = args.bound
        nh = args.module_size // 2
        self.fc1 = nn.Linear(dim_in, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))
        out = self.bound * F.tanh(out)
        return out


class NN3h(nn.Module):
    """Same as NN3B but use HardTanh to bound the output range. This cheaper and easier to train."""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN3h, self).__init__()
        assert args.module_size >= 3, 'module_size must be at least 3'
        self.f = afunc
        self.bound = args.bound
        nh = args.module_size // 2
        self.fc1 = nn.Linear(dim_in, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))
        out = F.hardtanh(out, min_val=-self.bound, max_val=self.bound)
        return out


class NN4(nn.Module):
    """Implements a standard 3 hidden layers neural net"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN4, self).__init__()
        assert args.module_size >= 3, 'module_size must be at least 3'
        self.f = afunc
        nh = args.module_size // 3
        self.fc1 = nn.Linear(dim_in, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, nh)
        self.fc4 = nn.Linear(nh, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc4(self.f(self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))))
        return out


class NN4h(nn.Module):
    """Same as NN3h but with 3 hidden layers neural net"""
    def __init__(self, dim_in, dim_out, args, afunc):
        super(NN4h, self).__init__()
        assert args.module_size >= 3, 'module_size must be at least 3'
        self.f = afunc
        self.bound = args.bound
        nh = args.module_size // 3
        self.fc1 = nn.Linear(dim_in, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, nh)
        self.fc4 = nn.Linear(nh, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc4(self.f(self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))))
        out = F.hardtanh(out, min_val=-self.bound, max_val=self.bound)
        return out


class Model:

    def __init__(self, train_loader, val_loader, criterion, val_criterion,
                 afunc, x_shape, dim_out, args, features_net=None):
        assert args.model in globals(), 'models/{}.py has no definition of model {}'.format(args.algorithm, args.model)
        dim_in = 1
        for size in x_shape: dim_in *= size
        self.net = globals()[args.model](dim_in, dim_out, args, afunc)
        self.args = args
        self.features_net = features_net
        self.optimizer = SGD(self.net, train_loader, val_loader, criterion, val_criterion,
                             args, transform_net=features_net)

    def train(self):
        """train the model and return the best (the one with smallest val_error)"""
        train_log = self.optimizer.train()
        best_train = self.optimizer.best_train
        best_train['module'] = 1
        best_validation = self.optimizer.best_validation
        best_validation['module'] = 1
        log = [{'module': 1, 'best_train': best_train, 'best_val': best_validation, 'log': train_log}]
        return best_train, best_validation, log

    def test(self, test_loader):
        """evaluate the best model on test data"""
        assert self.optimizer.best_validation['val_loss'] < float('inf'), "Model is not trained. Call train() first!"
        loss = 0
        mae = mse = mape = 0
        correct = 0
        for x, y in test_loader:
            x, y = x.to(self.optimizer.args.device), y.to(self.optimizer.args.device)
            if self.features_net is not None:
                x = self.features_net(x)
            output = self.optimizer.best_validation_model(x)
            yt = SGD.class2vec(y, output.size(1))  # form of y depends on loss type
            if self.optimizer.args.loss == 'nll':
                loss += self.optimizer.val_criterion(output, y).item() * y.size(0)  # sum up batch loss
            else:
                loss += self.optimizer.val_criterion(output, yt).item() * y.size(0)

            if self.optimizer.args.regression:
                mae += F.l1_loss(output, yt).item() * y.size(0)
                mse += F.mse_loss(output, yt).item() * y.size(0)
                if self.args.dataset == 'msd':  # y is shifted for msd dataset
                    mape += ((output - yt).abs() / (yt + 1998).abs()).sum().item()
                else:
                    mape += ((output - yt).abs() / yt.abs()).sum().item()

            if not self.optimizer.args.regression:  # classification task
                _, pred = output.max(1)
                correct += pred.eq(y).sum().item()
            elif self.optimizer.args.dataset.endswith("_r"):  # classification posed as regression [-1, 1]
                pred = output.new_ones(output.shape[0], dtype=y.dtype)
                pred[output[:, 0] < 0] = -1
                correct += pred.eq(y).sum().item()

        loss /= len(test_loader.dataset)
        mae /= len(test_loader.dataset)
        mse /= len(test_loader.dataset)
        rmse = mse**0.5
        error = 1.0 - correct / len(test_loader.dataset)
        return loss, error, rmse, mae

    @property
    def best_validation_model(self):
        return self.optimizer.best_validation_model

    @property
    def best_train_model(self):
        return self.optimizer.best_train_model

    @property
    def best_n_modules(self):  # number of modules/estimators in the final best model
        return 1
