"""
Implementation of features nets for vision dataset
"""

import torch.nn as nn
import torch
from optimizers.sgd import TorchSGD as SGD
import torch.nn.functional as F
from collections import OrderedDict
import math


class Linear(nn.Module):

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# ============ MNIST Features Net (Simple CNN)============================== #

class MnistFeatures(nn.Module):

    def __init__(self):
        super(MnistFeatures, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        if len(x.shape) == 3:
            x.unsqueeze_(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class MnistTFeatures(nn.Module):

    def __init__(self):
        super(MnistTFeatures, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        if len(x.shape) == 3:
            x.unsqueeze_(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.rule(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


def mnist():
    return nn.Sequential(MnistFeatures(), Linear(10 * 10 * 32, 10))


def mnist_t():
    return nn.Sequential(MnistTFeatures(), Linear(3 * 3 * 128, 10))


# ============ CIFAR10 Features Net (DenseNet-BC)============================== #
# based on github.com/junyuseu/pytorch-cifar-models.git

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_Cifar(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16),
                 num_init_features=24, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet_Cifar, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # initialize conv and bn parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class Cifar10F_342x4x4(nn.Module):

    def __init__(self):
        super(Cifar10F_342x4x4, self).__init__()

        growth_rate = 12
        block_config = (16, 16, 16)
        num_init_features = 24
        bn_size = 4
        drop_rate = 0
        num_classes = 10

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # initialize conv and bn parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


class Cifar10F_342x1x1(nn.Module):

    def __init__(self):
        super(Cifar10F_342x1x1, self).__init__()

        growth_rate = 12
        block_config = (16, 16, 16)
        num_init_features = 24
        bn_size = 4
        drop_rate = 0
        num_classes = 10

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # initialize conv and bn parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1)
        return out


def cifar10_5472f():
    return nn.Sequential(Cifar10F_342x4x4(), Linear(342*4*4, 10))


def cifar10_342f():
    return nn.Sequential(Cifar10F_342x1x1(), Linear(342, 10))


def densenet_cifar10():
    depth = 100
    k = 12
    N = (depth - 4) // 6
    model = DenseNet_Cifar(growth_rate=k, block_config=[N, N, N], num_init_features=2*k)
    return model


class Model:

    def __init__(self, train_loader, val_loader, criterion, val_criterion,
                 afunc, x_shape, n_classes, args, features_net=None):
        assert len(x_shape) == 3, 'Unexpected x_shape' + x_shape + '. This model only works with images dataset.'
        assert args.model in globals(), 'models/{}.py has no definition of model {}'.format(args.algorithm, args.model)
        assert afunc == F.relu, 'Unexpected activation. FeaturesNet is to be trained using ReLU only.'
        assert n_classes == 10, 'Unexpected input. FeaturesNet is to be trained for MNIST and CIFAR only.'
        assert features_net is None, 'features_net must ne None for features_net_test.py module!'
        self.net = globals()[args.model]()
        self.optimizer = SGD(self.net, train_loader, val_loader, criterion, val_criterion, args)

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
        assert self.optimizer.best_validation['val_error'] < 1., "Model is not trained yet. Call train() first!"
        loss = 0
        correct = 0
        for x, y in test_loader:
            x, y = x.to(self.optimizer.args.device), y.to(self.optimizer.args.device)
            output = self.optimizer.best_validation_model(x)
            yt = y if self.optimizer.args.loss == 'nll' else SGD.class2vec(y, output.size(1))  # form of y depends on loss type
            loss += self.optimizer.val_criterion(output, yt).item() * y.size(0)  # sum up batch loss
            if not self.optimizer.args.regression:
                _, pred = output.max(1)  # get the index of the max probability
                correct += pred.eq(y).sum().item()
        loss /= len(test_loader.dataset)
        error = 1.0 - correct / len(test_loader.dataset)
        return loss, error

    @property
    def best_validation_model(self):
        return self.optimizer.best_validation_model

    @property
    def best_train_model(self):
        return self.optimizer.best_train_model
