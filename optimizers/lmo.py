import copy
import torch.optim as optim
import math
from optimizers.scheduler import ReduceLROnPlateau


class LMO:
    """
    Implementation of Linear Minimization Oracle (LMO) for Frank Wolfe algorithm.
    Given fw_buffer containing batches of (x, y, z, z_grad, loss),
    optimize nn.Module net such that the inner product <grad, net(X)> is minimum
    """

    MAX_N_EPOCHS = 10**4

    def __init__(self, net, buffer, args, name=''):
        self.net = net
        self.buffer = buffer
        self.net.to(args.device)
        self.args = args
        self.name = name
        self.n_epochs = args.n_epochs
        self.epoch = 0

        self.best = {'loss': float('inf'), 'net': None, 'epoch': 0}

        params_groups = self.get_params_groups(net, args.no_bias_decay)
        if args.optimizer == 'sgd' or args.optimizer == 'nesterov':
            self.optimizer = optim.SGD(params_groups, lr=args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay, nesterov=(args.optimizer == 'nesterov'))
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(params_groups, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception('Unknown optimizer: ' + args.optimizer)

        if args.milestones:
            if args.milestones == [0]:
                milestones = self.get_auto_milestones(self.n_epochs, args.lr, args.min_lr, args.factor)
            else:
                milestones = args.milestones
            lr_decay_scheme = 'multi-step decay, gamma={}, milestones={}, min_lr={}'.format(
                args.factor, milestones, args.min_lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, args.factor)
        elif args.patience <= 0:
            gamma = args.factor if args.factor > 0 else (args.min_lr / args.lr)**(1./args.epochs)
            lr_decay_scheme = 'exponential decay each epoch, gamma={:.4f}, min_lr={}'.format(gamma, args.min_lr)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        else:
            lr_decay_scheme = 'decay on plateau, gamma={}, patience={}, min_lr={}, threshold={}.{}'.format(
                args.factor, args.patience, args.min_lr, args.threshold_mode, args.threshold)
            self.scheduler = ReduceLROnPlateau(self.optimizer, args.factor, args.patience,
                                               args.threshold, mode=args.threshold_mode)

        if args.verbose >= 6:
            print('   LMO optimizer loaded:\t{} ({}, lr={}, momentum={}, wdecay={},\n\t\t\t\t\t\t\t{})'.format(
                args.optimizer, 'regression' if args.regression else 'classification',
                args.lr, args.momentum, round(args.weight_decay, 9), lr_decay_scheme
            ))

    @staticmethod
    def get_auto_milestones(n_epochs, lr, min_lr, factor):
        """return milestones such that lr is equally reduce to min_lr in n_epochs"""
        k = int((math.log(min_lr) - math.log(lr) + 0.001) // math.log(factor) + 1)
        step_size = int((n_epochs - 1) // (k+1) + 1)
        milestones = [(i+1) * step_size for i in range(k)]
        return milestones

    @staticmethod
    def get_params_groups(net, no_bias_decay):
        """Separate parameters of the input nn.Module net into two groups, the bias group with no weight decay,
        and the group of remaining weights with weight decay specified by the weight-decay arguments"""
        group_decay = []
        group_no_decay = []

        for name, param in net.named_parameters():
            if param.requires_grad:
                if '.cw' in name or '.alpha' in name:
                    # parameterization of convex coefficients of wgn algorithm are not decayed
                    group_no_decay.append(param)
                elif no_bias_decay:
                    if '.bias' in name:
                        group_no_decay.append(param)
                    else:
                        group_decay.append(param)
                else:
                    group_decay.append(param)

        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def train_epoch(self):
        """train the next epoch"""
        self.epoch += 1
        train_loss = 0.
        n_trained = 0
        batch_log = []
        self.net.train()

        for batch_idx, (x, _, _, z_grad, _) in enumerate(self.buffer):

            self.optimizer.zero_grad()
            output = self.net(x)

            loss = output * z_grad  # inner product of this batch
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            n_trained += x.size(0)

            if math.isnan(train_loss):
                return train_loss, batch_log

            # batch_log: batchId, loss, error
            if self.args.verbose >= 7:
                # batch_log.append([batch_idx + 1, loss.item()])
                print('      lmo_train: epoch {} batch {:4} [{:7}/{} {:3}% ]   <grad, net>: {:.5f}'
                      .format(self.epoch, batch_idx + 1, n_trained, self.buffer.n_samples,
                              int(100 * n_trained / self.buffer.n_samples), loss.item()))

        return train_loss, batch_log

    def train(self):
        if self.n_epochs == 0:  # automatically determine the number of epochs
            self.n_epochs = self.MAX_N_EPOCHS
        log = []
        while self.epoch < self.n_epochs:

            train_loss, train_log = self.train_epoch()

            if math.isnan(train_loss):
                if self.args.verbose >= 6:
                    print('   lmo train_loss is NaN - probably due to high LR, try reduce it. Aborting LMO training!')
                log.append({'epoch': self.epoch, 'lr': lr, 'train_loss': train_loss, 'train_log': train_log})
                return log

            lr = round(self.optimizer.param_groups[0]['lr'], 7)
            if lr > self.args.min_lr:
                if type(self.scheduler) == ReduceLROnPlateau:
                    self.scheduler.step(train_loss, epoch=self.epoch)  # change learning rate according to scheduler
                else:
                    self.scheduler.step(epoch=self.epoch)
            elif self.n_epochs > self.epoch + self.args.patience:  # when lr <= min_lr, run maximally 10 more epochs
                self.n_epochs = self.epoch + self.args.patience

            if train_loss < self.best['loss']:
                self.best = {'loss': train_loss, 'epoch': self.epoch, 'net': copy.deepcopy(self.net)}

            if self.args.verbose >= 6:
                print('   lmo_train ({}):    epoch {}/{}:    <grad, net>: {:.5f}    lr: {}'.format(
                    self.name, self.epoch, self.n_epochs, train_loss, lr))

            log.append({'epoch': self.epoch, 'lr': lr, 'train_loss': train_loss, 'train_log': train_log})

        return log
