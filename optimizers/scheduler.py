from torch.optim import Optimizer
from bisect import bisect_right


class ReduceLROnPlateau:
    """
    Custom implementation of PyTorch lr_scheduler.ReduceLROnPlateau, which is invalid in rel mode if loss is negative!
    """

    def __init__(self, optimizer, factor=0.1, patience=10, threshold=1e-4, min_lr=0, mode='abs'):

        assert factor < 1.0, 'Factor should be < 1.0.'
        self.factor = factor
        assert isinstance(optimizer, Optimizer), '{} is not an Optimizer'.format(type(optimizer).__name__)
        self.optimizer = optimizer
        assert 0. < threshold < 1., '0 < threshold < 1 required.'
        self.threshold = threshold
        assert 0. <= min_lr <= 1., '0 <= min_lr <= 1 required.'
        self.min_lr = min_lr
        assert 0 < patience < 100, '0 < patience < 100 required.'
        self.patience = patience
        assert mode in {'rel', 'abs', 'adp'}, "mode must be 'rel' or 'abs' or 'adp'"
        self.mode = mode

        self.best = 1e12
        self.num_bad_epochs = 0

    def step(self, current_metric, epoch=None):
        if self._is_better(current_metric, self.best):
            self.best = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

        # print('current: {:.5f}   best: {:.5f}   bad epochs: {}'.format(
        #     current_metric, self.best, self.num_bad_epochs))

    def _is_better(self, current, best):
        if self.mode == 'rel':
            return current < best - self.threshold * abs(best)
        elif self.mode == 'abs':
            return current < best - self.threshold
        else:
            # adp=adaptive mode: try to balance between LMO and normal NN training.
            # if magnitude > 10 -> absolute mode, else relative mode
            magnitude = abs(best)
            if magnitude > 10.:  # abs mode
                return current < best - self.threshold
            else:  # rel mode
                return current < best - 1e-2 * self.threshold * magnitude

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > 1e-8:
                param_group['lr'] = new_lr


class CustomLR:
    """
    Custom implementation of PyTorch lr_scheduler.MultiStepLR, which does not support initial warming up LR!
    Example: milestones=[0, 1, 80, 120]  lrates=[0.01, 0.1, 0.01, 0.001]
    means that: lr 0.01 for epochs 0, lr 0.1 for epochs 1-79, lr 0.01 for epochs 80-119, lr 0.001 for epochs 120 up
    """

    def __init__(self, optimizer, milestones, lrates, last_epoch=0):
        if not isinstance(optimizer, Optimizer):
            raise ValueError('Invalid type for Optimizer . Got {}', type(optimizer).__name__)
        if list(milestones) != sorted(milestones) or milestones[0] != 0:
            raise ValueError('milestones should be a list of increasing integers starting with 0. Got {}', milestones)
        if not len(lrates) == len(milestones):
            raise ValueError('learningrates should have the same length as milestones. Got {}', lrates)
        self.optimizer = optimizer
        self.milestones = milestones
        self.lrates = lrates
        self.last_epoch = last_epoch
        self.step(last_epoch)

    def step(self, epoch=None):  # epoch=1 means epoch 1 is next (epoch 0 has been done)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch >= 0 else 0
        lr = self.lrates[bisect_right(self.milestones, self.last_epoch) - 1]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
