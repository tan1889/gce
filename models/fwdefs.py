"""
Various definitions to support models that use Frank Wolfe optimization method
"""

import gc
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def dim(shape):
    """Given shape, return the number of dimension"""
    d = 1
    for size in shape: d *= size
    return d


def get_size_of(obj):
    """For debugging: Return cuda and cpu memory footprint of an object (class, module, or tensor)"""
    cpu_bytes = cuda_bytes = 0
    if 'torch.nn.backends' in str(type(obj)):
        cpu_bytes = sys.getsizeof(obj)
    elif torch.is_tensor(obj):
        n_elements = 1;
        for dim in obj.size():
            n_elements *= dim
        size_in_bytes = n_elements * obj.element_size()
        if obj.is_cuda:
            cuda_bytes = size_in_bytes
        else:
            cpu_bytes = size_in_bytes
    elif not callable(obj) and hasattr(obj, 'data') and torch.is_tensor(obj.data):
        cpu_bytes, cuda_bytes = get_size_of(obj.data)
    elif type(obj) is list or type(obj) is tuple:
        for x in obj:
            cpu, cuda = get_size_of(x)
            cpu_bytes += cpu
            cuda_bytes += cuda
    elif type(obj) is dict or type(obj) is OrderedDict:
        for k, x in obj.items():
            cpu, cuda = get_size_of(x)
            cpu_bytes += cpu
            cuda_bytes += cuda
    elif hasattr(obj, '__dict__'):
        for k, x in obj.__dict__.items():
            cpu, cuda = get_size_of(x)
            cpu_bytes += cpu
            cuda_bytes += cuda
    else:
        cpu_bytes = sys.getsizeof(obj)  # normal cpu variable

    return cpu_bytes, cuda_bytes


def print_mem(obj, name=''):
    """For debugging: Print cuda and cpu memory footprint of an object (class, module, or tensor)"""
    if name == '' and hasattr(obj, '__name__'):
        name = obj.__name__
    cpu, cuda = get_size_of(obj)
    print("Memory footprint of object {}:   {:.2f}MB on cpu   {:.1f}MB on cuda".format(name, cpu/1e6, cuda/1e6))


def print_gc_tensors():
    """For debugging: Print cuda and cpu memory footprint of all tensors in the garbage collector"""
    cpu_bytes = cuda_bytes = 0
    cpu_count = cuda_count = 0
    cpu = dict()
    cuda = dict()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            key = ''
            n_elements = 1;
            for dim in obj.size():
                key += '{}x'.format(dim)
                n_elements *= dim
            size_in_bytes = n_elements * obj.element_size()
            if obj.is_cuda:
                cuda_bytes += size_in_bytes
                cuda_count += 1
                if key in cuda:
                    cuda[key]['count'] += 1
                    cuda[key]['size'] += size_in_bytes/1e6
                else:
                    cuda[key] = {'count': 1, 'size': size_in_bytes/1e6}
            else:
                cpu_bytes += size_in_bytes
                cpu_count += 1
                if key in cpu:
                    cpu[key]['count'] += 1
                    cpu[key]['size'] += size_in_bytes/1e6
                else:
                    cpu[key] = {'count': 1, 'size': size_in_bytes/1e6}

    if cpu_count > 0:
        print('CPU: {} TENSORS OCCUPYING {:.2f}MB'.format(cpu_count, cpu_bytes/1e6))
    for k in sorted(cpu.keys(), key=lambda x: cpu[x]['size'], reverse=True):
        print('{:4} tensors {:20} --> {:.2f}MB'.format(cpu[k]['count'], k[:-1], cpu[k]['size']))
    if cuda_count > 0:
        print('CUDA: {} TENSORS OCCUPYING {:.2f}MB'.format(cuda_count, cuda_bytes/1e6))
    for k in sorted(cuda.keys(), key=lambda x: cuda[x]['size'], reverse=True):
        print('{:4} tensors {:20} --> {:.2f}MB'.format(cuda[key]['count'], k[:-1], cuda[key]['size']))


class FixedModel(object):
    """This is the base class for simple fixed models, which are used to store and eval NN model, not for training."""
    def __init__(self):
        self._parameters = OrderedDict()

    def forward(self, *input, **kwargs):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def _apply_fn(self, v, fn):
        """Apply function fn to variable v if it is one of pytorch tensor / Parameter / Module.
        Else, if v is dict, tuple, list -> apply where its element is a tensor"""
        if hasattr(v, 'to') and hasattr(v, 'cpu') and hasattr(v, 'cuda'):
            v = fn(v)
        elif type(v) is list or type(v) is tuple:
            for x in v: x = self._apply_fn(x, fn)
        elif type(v) is dict:
            for k, x in v: v[k] = self._apply_fn(x, fn)
        return v

    def _apply(self, fn):
        for k, v in self.__dict__.items():
            setattr(self, k, self._apply_fn(v, fn))
        return self

    def cpu(self):
        return self._apply(lambda t: t.cpu())

    def cuda(self):
        return self._apply(lambda t: t.cuda())

    def to(self, *args, **kwargs):
        return self._apply(lambda t: t.to(*args, **kwargs))

    def state_dict(self):
        state = OrderedDict()
        for k, v in self.__dict__.items():
            state[k] = v
        return state

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __deepcopy__(self, memo):
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, copy.deepcopy(v, memo))
        return cpy


class Direction:
    r"""This module implements the direction in FW algorithms. Depending on direction_type we can have:
    Frank Wolfe direction FW, Away direction AW, or Pairwise direction PW."""

    def __init__(self, net, direction_type='FW', net2=None):
        assert direction_type in {'FW', 'AW', 'PW'}, 'Unknown direction type'
        self.net = net
        self.net2 = net2
        if net2 is not None: direction_type = 'PW'
        self.direction_type = direction_type

    def __call__(self, x, z):
        """x is batch input, z is sum_net(x), away direction: dA = z - net(x),
        Frank Wolfe direction: dFW = net(x) - z, Pairwise Frank Wolfe direction: d = net(x) - net2(x)"""
        out = self.net(x)
        if self.net2 is not None:  # Pairwise direction d
            out2 = self.net2(x)
            return out - out2
        elif self.direction_type == 'FW':  # Frank Wolfe direction dFW
            return out - z
        else:  # away direction, dA
            return z - out


class ConvexSum(FixedModel):
    r"""This module hold the convex sum of basic modules for the purpose of Frank Wolfe algorithm.
    It is not for training, only for eval. Each individual module is trained before being added here"""

    def __init__(self, first_module=None):
        super(ConvexSum, self).__init__()
        self.modules = self.cw = None
        if first_module is not None:
            self.modules = [self.freeze(first_module)]
            self.cw = torch.ones(1, 1, 1)  # initialize all convex weight to the first (zero) module

    def append(self, module, gamma, idx=None):
        """Add a trained module to the sum with convex weight = gamma, if idx is presented, this is Pairwise
        adding module step, meaning gamma mass is taken away from module idx to the newly added module"""
        self.modules.append(self.freeze(module))
        if idx is not None:  # Pairwise Frank Wolfe
            self.cw = torch.cat((self.cw, gamma * self.cw.new_ones((1, 1, 1))), dim=0)
            self.cw[idx] -= gamma  # take away the mass from the weight of neuron idx
            self.remove_if_zero(module_id=idx)
        else:  # Away-step Frank Wolfe
            self.cw = torch.cat(((1 - gamma) * self.cw, gamma * self.cw.new_ones((1, 1, 1))), dim=0)

    @staticmethod
    def freeze(module):
        """freeze parameters of a module and put it to eval() mode, so it will not be trained by optimizer"""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def find_away_direction(self, fw_buffer):
        r"""find the away direction, v_t = argmax_{v in Modules} <grad, v>,
        Returns: v_t, idx of v_t, alpha = convex coefficient of v_t"""
        assert self.modules, "No modules exist. Modules list should not be empty. Something is wrong!"
        p = self.cw.new(len(self.modules)).zero_()  # p stores <grad, v>
        for x, _, _, z_grad, _ in fw_buffer:
            outs = [net(x) for net in self.modules]
            outs = torch.stack(outs, dim=0)
            p_batch = z_grad * outs
            p_batch = p_batch.sum(2).sum(1)
            p += p_batch

        idx = p.max(0)[1].item()  # idx = arg max p = id of module with max inner product with z_grad
        alpha = self.cw[idx].item()  # convex coeff of module idx
        v = self.modules[idx]
        return v, idx, alpha

    def remove_if_zero(self, module_id):
        if self.cw[module_id] < 1e-7:  # convex coeff of module id is reduced to practically zero -> eliminate it
            # print('Removing module id {} as its convex-coeff={} is negligible'.format(
            #     module_id, self.cw[module_id].item()))
            del self.modules[module_id]
            self.cw = torch.cat((self.cw[:module_id], self.cw[module_id + 1:]), dim=0)

    def reduce_weight(self, idx, gamma):
        r"""reduce the convex coefficient of module idx by gamma, and increase other weight by gamma"""
        if gamma < 0 or gamma > self.cw[idx] / (1. - self.cw[idx]):
            print('Warning: gamma={} is out of range (cw[{}]={})'.format(gamma, idx, self.cw[idx]))
        self.cw *= (1. + gamma)
        self.cw[idx] -= gamma
        self.remove_if_zero(idx)

    def forward(self, x):
        outs = [net(x) for net in self.modules]
        outs = torch.stack(outs, dim=0)
        out = self.cw * outs
        out = out.sum(dim=0)
        return out


class Neuron(nn.Module):
    r"""This module implements the basic one neuron scaled by a bound: f(x) = B tanh(u^T x), where B \in R^+ is the
    bound of the output. This is for the basic greedy training with adding 1 neuron per 1 output dim at a time."""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        super(Neuron, self).__init__()
        self.dim_in = dim(shape_in)  # input dimension
        self.dim_out = dim_out  # output dimension
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        # init network weights
        self.weight = nn.Parameter(torch.zeros(self.dim_in, self.dim_out))
        self.bias = nn.Parameter(torch.zeros(self.dim_out))
        if not zero:
            std = 1. / (self.dim_in ** 0.5)
            self.weight.data.uniform_(-std, std)
            self.bias.data.uniform_(-std, std)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.addmm(self.bias, x, self.weight)
        x = self.bound * F.tanh(x)
        return x


class N2t(nn.Module):
    r"""This module implements a bounded neural network with two layers. For each output dimension, there is a NN
    B*tanh(v^T relu(Ux)), where U in R^{d x module_size}, v in {module_size x 1}. Note that the tanh unit in the
    last layer ensures the output is in (-1, 1) and then we scaled this by B. So, the output of each of this NN is
    in the interval (-B, B), and we have k such NN for k = dim_out. Each weight and neuron belongs to a specific
    output dimension and are not shared (in contrast to NN2Shared below).
    Number of weights: (dim_in+2)*dim_out*module_size + dim_out"""
    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use Neuron (much faster)'
        super(N2t, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # init network weights
        self.weight1 = nn.Parameter(torch.zeros(dim_out, self.dim_in, module_size))
        self.bias1 = nn.Parameter(torch.zeros(dim_out, 1, module_size))
        self.weight2 = nn.Parameter(torch.zeros(dim_out, module_size, 1))
        self.bias2 = nn.Parameter(torch.zeros(dim_out, 1, 1))
        if not zero:
            std = 1. / (self.dim_in ** 0.5)
            self.weight1.data.uniform_(-std, std)
            self.bias1.data.uniform_(-std, std)
            self.weight2.data.uniform_(-std, std)
            self.bias2.data.uniform_(-std, std)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.f(x.matmul(self.weight1) + self.bias1)
        out = self.bound * F.tanh(out.matmul(self.weight2) + self.bias2)
        out = out.squeeze(2).permute(1, 0)
        return out


class N2h(nn.Module):
    r"""Same as N2t above, but using hardtanh instead of tanh to bound the function output, hardtanh is clamping.
    It is cheaper and easier to train."""
    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use Neuron (much faster)'
        super(N2h, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # init network weights
        self.weight1 = nn.Parameter(torch.zeros(dim_out, self.dim_in, module_size))
        self.bias1 = nn.Parameter(torch.zeros(dim_out, 1, module_size))
        self.weight2 = nn.Parameter(torch.zeros(dim_out, module_size, 1))
        self.bias2 = nn.Parameter(torch.zeros(dim_out, 1, 1))
        if not zero:
            std = 1. / (self.dim_in ** 0.5)
            self.weight1.data.uniform_(-std, std)
            self.bias1.data.uniform_(-std, std)
            self.weight2.data.uniform_(-std, std)
            self.bias2.data.uniform_(-std, std)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.f(x.matmul(self.weight1) + self.bias1)
        out = out.matmul(self.weight2) + self.bias2
        out = F.hardtanh(out, min_val=-self.bound, max_val=self.bound)
        out = out.squeeze(2).permute(1, 0)
        return out


class N2DR(nn.Module):
    r"""Same as NN2 but with dropout"""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False, xavier_init=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use Neuron (much faster)'
        super(N2DR, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # init network weights
        self.weight1 = nn.Parameter(torch.zeros(dim_out, self.dim_in, module_size))
        self.bias1 = nn.Parameter(torch.zeros(dim_out, 1, module_size))
        self.weight2 = nn.Parameter(torch.zeros(dim_out, module_size, 1))
        self.bias2 = nn.Parameter(torch.zeros(dim_out, 1, 1))
        self.dr = nn.Dropout()
        if not zero:
            std = 1. / (self.dim_in ** 0.5)
            if xavier_init: std = 6**0.5 / (self.dim_in + module_size)
            self.weight1.data.uniform_(-std, std)
            if not xavier_init: self.bias1.data.uniform_(-std, std)  # for xavier init, bias=0
            if xavier_init: std = 6**0.5 / (module_size + 1)
            self.weight2.data.uniform_(-std, std)
            if not xavier_init: self.bias2.data.uniform_(-std, std)  # for xavier init, bias=0

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.dr(self.f(x.matmul(self.weight1) + self.bias1))
        out = self.bound * F.tanh(out.matmul(self.weight2) + self.bias2)
        out = out.squeeze(2).permute(1, 0)
        return out


class N2CW(nn.Module):
    r"""Same as NN2 above, but we use convex coefficients to bound the output of the network"""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False, xavier_init=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use Neuron (much faster)'
        super(N2CW, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # init network weights
        self.weight1 = nn.Parameter(torch.zeros(dim_out, self.dim_in, module_size))
        self.bias1 = nn.Parameter(torch.zeros(dim_out, 1, module_size))
        self.weight2 = nn.Parameter(torch.zeros(dim_out, module_size, 1))
        self.bias2 = nn.Parameter(torch.zeros(dim_out, 1, 1))
        self.bias2neg = nn.Parameter(torch.zeros(dim_out, 1, 1))
        if not zero:
            std = 1. / (self.dim_in ** 0.5)
            if xavier_init: std = 6**0.5 / (self.dim_in + module_size)
            self.weight1.data.uniform_(-std, std)
            if not xavier_init: self.bias1.data.uniform_(-std, std)
            self.weight2.data.fill_(0.05)
            self.bias2.data.fill_(0.05)
            self.bias2neg.data.fill_(0.05)

    @staticmethod
    def normalize_cw(W, b1, b2):
        """W is module_size x dim_out, each column k of this + bias b1[k] + b2[k] must be convex coefficients (sum
        to 1)"""
        A = W.abs()  # normalize using 1/K + |cw_i| / (K + sum(cw))
        sums = A.sum(dim=1) + b1.abs() + b2.abs()
        k = A.size(1) + 2
        W = (1. / k + A) / (1. + sums)
        b1 = (1. / k + b1.abs()) / (1. + sums)
        b2 = (1. / k + b2.abs()) / (1. + sums)
        return W, b1, b2

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.f(x.matmul(self.weight1) + self.bias1)
        W, b1, b2 = self.normalize_cw(self.weight2, self.bias2, self.bias2neg)
        out = self.bound * (out.matmul(W) + b1 - b2)
        out = out.squeeze(2).permute(1, 0)
        return out


class N3h(nn.Module):
    r"""Same as N2h but with 2 hidden layers"""
    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size > 2, 'module_size must be integer >= 3. For module_size = 1 use Neuron (much faster)'
        super(N3h, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # init network weights
        self.weight1 = nn.Parameter(torch.zeros(dim_out, self.dim_in, module_size))
        self.bias1 = nn.Parameter(torch.zeros(dim_out, 1, module_size))
        self.weight2 = nn.Parameter(torch.zeros(dim_out, module_size, module_size))
        self.bias2 = nn.Parameter(torch.zeros(dim_out, 1, module_size))
        self.weight3 = nn.Parameter(torch.zeros(dim_out, module_size, 1))
        self.bias3 = nn.Parameter(torch.zeros(dim_out, 1, 1))
        if not zero:
            std = 1. / (self.dim_in ** 0.5)
            self.weight1.data.uniform_(-std, std)
            self.bias1.data.uniform_(-std, std)
            self.weight2.data.uniform_(-std, std)
            self.bias2.data.uniform_(-std, std)
            self.weight3.data.uniform_(-std, std)
            self.bias3.data.uniform_(-std, std)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.f(x.matmul(self.weight1) + self.bias1)
        x = self.f(x.matmul(self.weight2) + self.bias2)
        x = x.matmul(self.weight3) + self.bias3
        x = F.hardtanh(x, min_val=-self.bound, max_val=self.bound)
        x = x.squeeze(2).permute(1, 0)
        return x


class SN2t(nn.Module):
    r"""Same as the NN2 module above, but here the weight and neurons (features) are shared between output dimension.
    So, instead of having dim_out NN2 nets, we only have one NN2 net where features are shared between dimension.
    Specifically, the module is B afunc(V afunc(Ux)). So, this module is like a normal NN with 1 hidden layer,
    although scaling and afunc is applied to the NN output layer so that it is bounded in (-B, B). Here, we assume
    that afunc is in (-1, 1).
    Number of weights of this module is: dim_in*module_size + module_size + module_size*dim_out + dim_out"""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use module Neuron (much faster)'
        super(SN2t, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # init network weights
        self.fc1 = nn.Linear(self.dim_in, module_size)
        self.fc2 = nn.Linear(module_size, dim_out)
        if zero:
            self.fc2.weight.data.zero_()
            self.fc2.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc2(self.f(self.fc1(x)))
        out = self.bound * F.tanh(out)
        return out


class SN2h(nn.Module):
    r"""Same as the NN2Shared but using hardtanh at output. This is better and easier to train."""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size >= 1, 'module_size must be integer >= 1. For module_size = 1 use module Neuron (much faster)'
        super(SN2h, self).__init__()
        self.dim_in = dim(shape_in)
        self.dim_out = dim_out
        self.module_size = module_size
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        self.fc1 = nn.Linear(self.dim_in, module_size)
        self.fc2 = nn.Linear(module_size, dim_out)
        if zero:
            self.fc2.weight.data.zero_()
            self.fc2.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc2(self.f(self.fc1(x)))
        out = F.hardtanh(out, min_val=-self.bound, max_val=self.bound)
        return out


class SN3h(nn.Module):
    r"""Same as the SN3t but using hardtanh at output"""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use module Neuron (much faster)'
        super(SN3h, self).__init__()
        dim_in = dim(shape_in)
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        self.fc1 = nn.Linear(dim_in,  module_size)
        self.fc2 = nn.Linear(module_size, module_size)
        self.fc3 = nn.Linear(module_size, dim_out)
        if zero:
            self.fc3.weight.data.zero_()
            self.fc3.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))
        out = F.hardtanh(out, min_val=-self.bound, max_val=self.bound)
        return out


class SN4h(nn.Module):
    r"""Same as the NN3Shared but using hardtanh at output"""

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert module_size > 1, 'module_size must be integer >= 2. For module_size = 1 use module Neuron (much faster)'
        super(SN4h, self).__init__()
        dim_in = dim(shape_in)
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        self.fc1 = nn.Linear(dim_in,  module_size)
        self.fc2 = nn.Linear(module_size, module_size)
        self.fc3 = nn.Linear(module_size, module_size)
        self.fc4 = nn.Linear(module_size, dim_out)
        if zero:
            self.fc4.weight.data.zero_()
            self.fc4.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc4(self.f(self.fc3(self.f(self.fc2(self.f(self.fc1(x)))))))
        out = F.hardtanh(out, min_val=-self.bound, max_val=self.bound)
        return out


class CNN2(nn.Module):

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert len(shape_in) == 3 and dim_out > 1, 'This module can only be used for image classification'
        super(CNN2, self).__init__()
        d, h, w = shape_in
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # cnn layers
        self.conv1 = nn.Conv2d(d, 32, kernel_size=5, bias=False)  # s - 4
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, bias=False)  # (s - 9)/2 + 1
        self.bn2 = nn.BatchNorm2d(32)
        # fc layers
        size = lambda s: (s - 9) // 2  + 1
        fc1_size = size(h) * size(w) * 32
        self.fc1 = nn.Linear(fc1_size, module_size)
        self.fc2 = nn.Linear(module_size, dim_out)
        if zero:
            self.fc2.weight.data.zero_()
            self.fc2.bias.data.zero_()

    def forward(self, x):
        x = self.f(self.bn1(self.conv1(x)))
        x = self.f(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        out = self.bound * F.tanh(self.fc2(self.f(self.fc1(x))))  # hidden layer
        return out


class CNN3(nn.Module):

    def __init__(self, shape_in, dim_out, module_size, bound, afunc, zero=False):
        assert len(shape_in) == 3 and dim_out > 1, 'This module can only be used for image classification'
        super(CNN3, self).__init__()
        d, h, w = shape_in
        self.bound = bound  # as in B * tanh(w^T x), scaling factor to the output of activation function
        self.f = afunc
        # cnn layers
        self.conv1 = nn.Conv2d(d, 32, kernel_size=5, bias=False)  # s - 4
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, bias=False)  # (s - 9)/2 + 1
        self.bn2 = nn.BatchNorm2d(32)
        # fc layers
        size = lambda s: (s - 9) // 2  + 1
        fc1_size = size(h) * size(w) * 32
        self.fc1 = nn.Linear(fc1_size, 2 * module_size//3)
        self.fc2 = nn.Linear(2 * module_size//3, module_size//3)
        self.fc3 = nn.Linear(module_size//3, dim_out)
        if zero:
            self.fc3.weight.data.zero_()
            self.fc3.bias.data.zero_()

    def forward(self, x):
        x = self.f(self.bn1(self.conv1(x)))
        x = self.f(self.bn2(self.conv2(x)))
        x = self.f(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        out = self.bound * F.tanh(self.fc3(self.f(self.fc2(self.f(self.fc1(x))))))  # hidden layer
        return out
