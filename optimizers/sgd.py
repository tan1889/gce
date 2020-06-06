import copy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
from optimizers.buffer import DataLoaderBuffer
from optimizers.scheduler import ReduceLROnPlateau


class TorchSGD:
    r"""
    Implements a stochastic gradient descent optimizer, which serves as the default optimizer for models defined
    in /models (apart from Frank Wolfe models, which use lmp.py).

    Arguments:
        net: the model to be trained. This must be a subclass of nn.Module
        train_loader: data loader for the training dataset
        val_loader: data loader for the validation dataset
        criterion: loss function (nll or mse or l1)
        val_criterion: For regression task, use this option to set different loss for training and validation.
            E.g. train loss is mse, but l1 could be used for validation and showing result.
        args: training parameters passed in from run_nn_gce.py
        name: name of this instance, which serves as prefix to the text printed out
        sum_net: for wgnet algorithm, this is the convex combination of modules that has been trained,
            the output of this net will be combined with the output of the module to be trained.
        transform_net: this can be either a features extractor or the trained modules in the case of dgn.
            The output of this net is used to train the module that is to be trained.
        alpha: alpha=0 indicates dgn training is required (no alpha involved. The input goes through
            'trained_nets', then goes to 'net' (we only optimize 'net'). alpha > 0 indicates wgn training,
            the output is convex combination of outputs of all nets: out = alpha * net(x) + (1-alpha) * trained_nets(x)
    """
    MAX_N_EPOCHS = 10**4

    def __init__(self, net, train_loader, val_loader, criterion, val_criterion,
                 args, name='', sum_net=None, transform_net=None, alpha=0.):
        self.net = net
        self.net.to(args.device)
        self.args = args
        self.name = name
        self.alpha = alpha
        self.n_epochs = args.n_epochs
        self.epoch = 0

        # batch_size is specified for warm_start training
        self.train_buffer = DataLoaderBuffer(train_loader, args.caching, args.device, sum_net, transform_net)
        val_caching = 3 if args.caching else 0
        self.val_buffer = DataLoaderBuffer(val_loader, val_caching, args.device, sum_net, transform_net)

        self.best_train = {'train_loss': float('inf'), 'train_error': float('inf'),
                           'train_rmse': float('inf'), 'train_mae': float('inf'),
                           'val_loss': float('inf'), 'val_error': float('inf'),
                           'val_rmse': float('inf'), 'val_mae': float('inf'),
                           'epoch': 0}
        self.best_train_model = None
        self.best_validation = {'train_loss': float('inf'), 'train_error': float('inf'),
                                'val_loss': float('inf'), 'val_error': float('inf'), 'epoch': 0}
        self.best_validation_model = None

        self.criterion = criterion
        self.val_criterion = val_criterion

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
                milestones = self.get_auto_milestones(self.n_epochs, args.lr, args.min_lr, args.factor, )
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

        if args.verbose >= 5:
            self.print_interval = 1
        elif args.verbose <= 3:
            self.print_interval = 10**10  # will print only final result
        else:  # args.verbose == 4
            self.print_interval = max(10, self.n_epochs // 100)  # print module result 10 times
            if self.n_epochs <= 0:  # unlimited number of epochs
                self.print_interval = 100
        if args.verbose >= 3:
            print('SGD OPTIMIZER LOADED:\t{} ({}, bsize={}, lr={}, momentum={}, weight_decay={},\n\t\t\t{})\n'.format(
                args.optimizer, 'regression' if args.regression else 'classification',
                args.batch_size, args.lr, args.momentum, round(args.weight_decay, 9), lr_decay_scheme
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
        and the group of remaining weights with weight decay specified by the weight-decay argument"""
        group_decay = []
        group_no_decay = []

        for name, param in net.named_parameters():
            if param.requires_grad:
                if '.cw' in name or '.alpha' in name:  # parameterization of convex coeffs of wgn algorithm are not decayed
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

    @staticmethod
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

    def train_epoch(self):
        """train the next epoch"""
        self.epoch += 1
        train_loss = 0.
        mse = mae = 0.
        n_correct = 0
        n_trained = 0
        n_data = self.train_buffer.n_samples
        batch_log = []
        self.net.train()
        # get batch data from train_buffer. x,y is the data point/label, s is the output of trained_nets,
        # which is convex combination of trained modules for wgnet, or output of trained layers for dgnet,
        # or it can be a net for features extraction.
        for batch_idx, (x, y, s) in enumerate(self.train_buffer):

            self.optimizer.zero_grad()
            output = self.net(x)

            if s is not None:  # s, if not None, is output of sum_net
                output = self.alpha * output + (1. - self.alpha) * s

            yt = y if self.args.loss == 'nll' else self.class2vec(y, output.size(1))  # form of y depends on loss type

            loss = self.criterion(output, yt)
            loss.backward()
            self.optimizer.step()

            m = n_trained
            n_trained += y.shape[0]
            train_loss = (m / n_trained) * train_loss + (y.shape[0]/n_trained) * loss.item()
            if math.isnan(train_loss):
                return float('nan'), float('nan'), float('nan'), float('nan'), []

            if self.args.regression:
                mae += F.l1_loss(output, yt).item() * y.size(0)
                mse += F.mse_loss(output, yt).item() * y.size(0)

            n_ok = 0

            if not self.args.regression:   # classification task
                _, pred = output.max(1)
                n_ok = pred.eq(y).sum().item()
            elif self.args.dataset.endswith("_r"):  # classification posed as regression [-1, 1]
                pred = output.new_ones(output.shape[0], dtype=y.dtype)
                pred[output[:, 0] < 0] = -1
                n_ok = pred.eq(y).sum().item()

            n_correct += n_ok

            # batch_log: batchId, loss, error
            if self.args.verbose > 2:  # saving batch_log is memory intensive, only do so if verbose > 2
                batch_log.append([batch_idx + 1, loss.item(),
                                  (self.train_buffer.batch_size - n_ok)/self.train_buffer.batch_size])
            if self.args.verbose >= 7:
                if self.args.regression and not self.args.dataset.endswith("_r"):
                    print('   train (epoch {})   batch {:4} [{:7}/{} {:3}% ]   loss: {:.5f}'
                          .format(self.epoch, batch_idx + 1, n_trained, n_data, int(100. * n_trained / n_data),
                                  batch_log[-1][1]))
                else:
                    print('   train (epoch {})   batch {:4} [{:7}/{} {:3}% ]   loss: {:.5f}   accuracy: {:.2f}%'
                          .format(self.epoch, batch_idx + 1, n_trained, n_data, int(100. * n_trained / n_data),
                                  batch_log[-1][1], 100.*(1. - batch_log[-1][2])))

        train_error = 1. - n_correct / n_trained
        mse /= n_trained
        rmse = mse**0.5
        mae /= n_trained

        if self.epoch % self.print_interval == 0 or (self.args.verbose >= 4 and self.epoch == self.n_epochs):
            if self.args.regression and not self.args.dataset.endswith("_r"):
                print('TRAIN {} Epoch {}/{}:   Loss: {:.5f}'.format(
                    self.name, self.epoch, self.n_epochs, train_loss
                ))
            else:
                print('TRAIN {} Epoch {}/{}:   Loss: {:.5f}   Accuracy: {:.2f}% ({}/{})'.format(
                    self.name, self.epoch, self.n_epochs, train_loss, 100*n_correct/n_trained, n_correct, n_trained
                ))
        return train_loss, train_error, rmse, mae, batch_log

    def eval(self, buffer):
        self.net.eval()
        eval_loss = 0.
        correct = n_eval = 0
        mae = mse = 0
        for x, y, s in buffer:

            output = self.net(x)

            if s is not None:  # s, if not None, is output of sum_net
                output = self.alpha * output + (1. - self.alpha) * s

            yt = y if type(self.val_criterion) is nn.CrossEntropyLoss else self.class2vec(y, output.size(1))

            loss = self.val_criterion(output, yt)

            n_eval += y.shape[0]  # y.shape[0] = number of samples in this batch. Next, update loss incrementally
            eval_loss = ((n_eval - y.shape[0])/n_eval) * eval_loss + (y.shape[0]/n_eval) * loss.item()

            if self.args.regression:
                mae += F.l1_loss(output, yt).item() * y.size(0)
                mse += F.mse_loss(output, yt).item() * y.size(0)

            if not self.args.regression:  # classification task
                _, pred = output.max(1)
                correct += pred.eq(y).sum().item()
            elif self.args.dataset.endswith("_r"):  # classification posed as regression [-1, 1]
                pred = output.new_ones(output.shape[0], dtype=y.dtype)
                pred[output[:, 0] < 0] = -1
                correct += pred.eq(y).sum().item()

        mae /= n_eval
        mse /= n_eval
        rmse = mse ** 0.5
        error = 1.0 - correct / n_eval
        return eval_loss, error, rmse, mae

    def validate_epoch(self):
        loss, error, rmse, mae = self.eval(self.val_buffer)
        lr = round(self.optimizer.param_groups[0]['lr'], 7)
        if self.epoch % self.print_interval == 0:
            if self.args.regression and not self.args.dataset.endswith("_r"):
                print('VALDT {} Epoch {}/{}:   Loss: {:.5f}   LR: {}\n'.format(
                    self.name, self.epoch, self.n_epochs, loss, lr
                ))
            else:
                print('VALDT {} Epoch {}/{}:   Loss: {:.5f}   Accuracy: {:.2f}% ({:.0f}/{})   LR: {}\n'.format(
                    self.name, self.epoch, self.n_epochs, loss, 100. * (1.0 - error),
                    (1.0 - error) * self.val_buffer.n_samples, self.val_buffer.n_samples, lr
                ))
        return loss, error, rmse, mae, lr

    def train(self, n_epochs=0):
        if n_epochs > 0:
            self.n_epochs = n_epochs
        if self.n_epochs == 0:  # automatically determine the number of epochs
            self.n_epochs = self.MAX_N_EPOCHS

        log = []
        while self.epoch < self.n_epochs:

            alpha = 0.
            if hasattr(self.net, 'cw'):
                cw, alpha = self.net.normalize_cw(self.net.cw, self.net.alpha)
                # print('convex coeffs: alpha={}, cw={}'.format(round(alpha.data.item(),4),
                #                                               [round(w.item(),4) for w in cw.data]))

            tr_loss, tr_error, tr_rmse, tr_mae, train_log = self.train_epoch()

            assert not math.isnan(tr_loss), 'train_loss is NaN (probably too high LR, reduce it!)'

            vl_loss, vl_error, vl_rmse, vl_mae, lr = self.validate_epoch()

            result = {'module': 1, 'epoch': self.epoch,
                      'train_loss': tr_loss, 'train_error': tr_error, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
                      'val_loss': vl_loss, 'val_error': vl_error, 'val_rmse': vl_rmse, 'val_mae': vl_mae}

            if lr > self.args.min_lr:
                if type(self.scheduler) == ReduceLROnPlateau:
                    self.scheduler.step(tr_loss, epoch=self.epoch)  # change learning rate according to scheduler
                else:
                    self.scheduler.step(epoch=self.epoch)
            elif self.n_epochs > self.epoch + self.args.patience:  # when lr <= min_lr, run maximally 10 more epochs
                self.n_epochs = self.epoch + self.args.patience

            if tr_loss < self.best_train['train_loss']:
                self.best_train = result.copy()
                self.best_train_model = copy.deepcopy(self.net)

            if (self.args.regression and vl_loss < self.best_validation['val_loss']) \
                    or (not self.args.regression and vl_error < self.best_validation['val_error']):
                self.best_validation = result.copy()
                self.best_validation_model = copy.deepcopy(self.net)

            log.append({'epoch': self.epoch, 'lr': lr, 'train_loss': tr_loss, 'train_error': tr_error,
                        'val_loss': vl_loss, 'val_error': vl_error, 'alpha': alpha, 'train_log': train_log})

            if self.epoch % self.print_interval == 0 or self.epoch >= self.n_epochs:
                if self.args.regression and not self.args.dataset.endswith("_r"):
                    print('BEST TRAIN RESULT:   Loss: {:.5f}   (Epoch {})'.format(
                        self.best_train['train_loss'], self.best_train['epoch']
                    ))
                    print('BEST VALDT RESULT:   Loss: {:.5f}   (Epoch {})\n'.format(
                        self.best_validation['val_loss'], self.best_validation['epoch']
                    ))
                else:
                    print('BEST TRAIN RESULT:   Loss: {:.5f}   Accuracy: {:.2f}%   (Epoch {})'.format(
                        self.best_train['train_loss'], 100. * (1 - self.best_train['train_error']),
                        self.best_train['epoch']
                    ))
                    print('BEST VALDT RESULT:   Loss: {:.5f}   Accuracy: {:.2f}%   (Epoch {})\n'.format(
                        self.best_validation['val_loss'], 100. * (1 - self.best_validation['val_error']),
                        self.best_validation['epoch']
                    ))

        return log
