from optimizers.lmo import LMO
from optimizers.buffer import DataLoaderBufferFW
from models.fwdefs import *
from optimizers.linesearch import *
from optimizers.sgd import TorchSGD as SGD
from config import *
import copy
from launcher import get_criterion
import time


class Model:
    """
    Implementation of Pairwise-step Frank Wolfe to optimize neural network greedily
    """

    def __init__(self, train_loader, val_loader, criterion, val_criterion, afunc,
                 shape_in, dim_out, args, features_net=None):
        assert args.model in globals(), 'models/{}.py has no definition of model {}'.format(args.algorithm, args.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.criterion.reduction = 'none'  # required for caching loss of individual batch sample
        self.val_criterion = val_criterion
        self.val_criterion.reduction = 'none'
        self.features_net = features_net
        self.args = args
        self.afunc = afunc
        self.dim_out = dim_out
        self.shape_in = shape_in
        self.BasicModule = globals()[args.model]

        if not args.warm_start:
            first_module = self.BasicModule(self.shape_in, dim_out, args.module_size, args.bound, afunc, zero=True)
        else:
            # warm start with a first module trained by sgd, correct ws parameters of args
            if args.verbose >= 3:
                print('TRAINING WARM START MODULE USING SGD...')
            ws_args = copy.deepcopy(args)
            ws_args.optimizer = args.ws_optimizer = get_ws_param(args, 'optimizer')
            ws_args.weight_decay = args.ws_weight_decay = get_ws_param(args, 'weight_decay')
            ws_args.batch_size = args.ws_batch_size = get_ws_param(args, 'batch_size')
            ws_args.lr = args.ws_lr = get_ws_param(args, 'lr')
            ws_args.patience = args.ws_patience = get_ws_param(args, 'patience')
            ws_args.threshold = args.ws_threshold = get_ws_param(args, 'threshold')
            ws_args.threshold_mode = args.ws_threshold_mode = get_ws_param(args, 'threshold_mode')
            ws_args.min_lr = args.ws_min_lr = get_ws_param(args, 'min_lr')
            ws_args.milestones = args.ws_milestones = get_ws_param(args, 'milestones')
            ws_args.loss = args.ws_loss = get_ws_param(args, 'loss')
            ws_args.validation_loss = args.ws_validation_loss = get_ws_param(args, 'validation_loss')

            ws_criterion = criterion
            ws_val_criterion = val_criterion
            if ws_args.loss != args.loss:
                ws_criterion = get_criterion(ws_args.loss, ws_args.regression)
            if ws_args.validation_loss != args.validation_loss:
                ws_val_criterion = get_criterion(ws_args.validation_loss, ws_args.regression)

            first_module = self.BasicModule(self.shape_in, dim_out, ws_args.module_size, ws_args.bound, afunc)
            sgd = SGD(first_module, train_loader, val_loader,
                      ws_criterion, ws_val_criterion, ws_args, transform_net=features_net)
            sgd.train()
            first_module = sgd.best_validation_model
            if args.verbose >= 3:
                print('WARM START MODULE TRAINED:    {} epochs    '
                      'val_loss:min={:.5f} at ep {}    train_loss:min={:.5f} at ep {}\n'.format(
                        sgd.epoch, sgd.best_validation['val_loss'], sgd.best_validation['epoch'],
                        sgd.best_train['train_loss'], sgd.best_train['epoch']))

        self.prev_model_size = 0
        self.sum_net = ConvexSum(first_module)
        if hasattr(args, 'filename') and args.filename:
            import os.path
            assert os.path.isfile(args.filename), 'Specified checkpoint file does not exists!'
            prev_state = torch.load(args.filename)
            self.sum_net.load_state_dict(prev_state)
            self.prev_model_size = len(self.sum_net.modules)
            if args.verbose >= 3:
                print('Loaded checkpoint model ({} modules): {}'.format(self.prev_model_size, args.filename))

        if args.model == 'Neuron':
            args.module_size = 1        # change args to match parameter of a Neuron
            args.activation = 'tanh'

        self.sum_net.to(args.device)

        self.best_train = {'module': 0, 'epoch': 0, 'train_loss': float('inf'), 'train_error': float('inf'),
                           'val_loss': float('inf'), 'val_error': float('inf')}
        self.best_train_model = None
        self.best_validation = {'module': 0, 'epoch': 0, 'train_loss': float('inf'), 'train_error': float('inf'),
                                'val_loss': float('inf'), 'val_error': float('inf')}
        self.best_validation_model = None

    def train(self):
        # set factor to decrease weight decay after each module
        factor = 1.
        wd0 = self.args.weight_decay
        wd1 = self.args.min_weight_decay
        if self.args.n_modules > 1 and wd1 > 0 and wd0 > 0:
            factor = (wd1 / wd0)**(1 / (self.args.n_modules-1))

        log = []
        train_log = []
        epoch = gamma = idx = -1
        lmo_loss = lmo_epochs = 0

        train_buffer = DataLoaderBufferFW(self.train_loader, self.args.caching, self.args.device, self.criterion,
                                          self.dim_out, transform_net=self.features_net)
        val_caching = 4 if self.args.caching else 0
        val_buffer = DataLoaderBufferFW(self.val_loader, val_caching, self.args.device, self.val_criterion,
                                        self.dim_out, transform_net=self.features_net, requires_grad=False)

        if self.args.verbose >= 5:
            print_interval = 1
        elif self.args.verbose <= 3:
            print_interval = 10**10  # will print only final result
        else:  # args.verbose == 4
            print_interval = max(1, (self.args.n_modules + 1 - self.prev_model_size) // 10)  # print 10 results

        time0 = time.time()

        for t in range(self.prev_model_size, self.args.n_modules + 1):

            train_buffer.update(self.sum_net)
            val_buffer.update(self.sum_net)

            tr_loss, tr_error, tr_rmse, tr_mae = self.eval(train_buffer)
            vl_loss, vl_error, vl_rmse, vl_mae = self.eval(val_buffer)
            result = {'module': t, 'epoch': epoch, 'runtime': time.time() - time0, 'gamma': gamma, 'away_id': idx,
                      'train_loss': tr_loss, 'train_error': tr_error, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
                      'val_loss': vl_loss, 'val_error': vl_error, 'val_rmse': vl_rmse, 'val_mae': vl_mae}

            # update self.best_train, which stores the model with min train loss
            if tr_loss < self.best_train['train_loss']:
                self.best_train = result.copy()
                self.best_train_model = copy.deepcopy(self.sum_net)

                if self.args.save_result > 2:  # save checkpoint if args.save_result = 2
                    model = copy.deepcopy(self.sum_net)
                    state_dict = model.cpu().state_dict()
                    fname = config['DEFAULT']['checkpoints'] + 'pfw_{}_model_checkpoint_t{}.pt'.format(self.args.dataset, t)
                    torch.save(state_dict, fname)
                    if self.args.verbose >= 5:
                        print('   saved checkpoint:', fname)
                    del model

            # update self.best_validation, which stores the model with min validation loss/error
            if (self.args.regression and vl_loss + 1e-7 < self.best_validation['val_loss']) or \
                    (not self.args.regression and vl_error + 1e-6 < self.best_validation['val_error']):
                self.best_validation = result.copy()
                self.best_validation_model = copy.deepcopy(self.sum_net)

            log.append(result)
            log[-1]['log'] = train_log

            early_stopping = self.best_validation_model and t >= self.args.min_n_modules \
                and (t - self.best_validation['module'] >= self.args.early_stopping > 0
                     or t - self.best_train['module'] >= 10 or tr_loss < 1e-9)

            if self.args.verbose >= 3 and (t == self.prev_model_size or t == self.args.n_modules
                                           or t % print_interval == 0 or early_stopping):

                print('   Pairwise Frank Wolfe step:  gamma={}  away_id={}'.format(round(gamma, 5), idx))

                if self.args.regression and not self.args.dataset.endswith("_r"):
                    print('MODULE #{}/{}:  {:.2f}s  train_loss: {:.5f}  val_loss: {:.5f}   LMO: epochs={} loss= {:.4f}'
                          .format(t, self.args.n_modules, time.time() - time0, tr_loss, vl_loss, lmo_epochs, lmo_loss))
                    print('        TRAIN:  RMSE = {:.5f}     MAE = {:.5f}'.format(tr_rmse, tr_mae))
                    print('          VAL:  RMSE = {:.5f}     MAE = {:.5f}'.format(vl_rmse, vl_mae))
                    print('CURRENT BEST TRAIN RESULT:   Loss: {:.5f}  @  Module/Epoch: {}/{}'
                          .format(self.best_train['train_loss'], self.best_train['module'], self.best_train['epoch']))
                    print('CURRENT BEST VALDT RESULT:   Loss: {:.5f}  @  Module/Epoch: {}/{}\n'
                          .format(self.best_validation['val_loss'],
                                  self.best_validation['module'], self.best_validation['epoch']))
                else:
                    print('MODULE #{}/{}:  {:.2f}s  loss: {:.5f}  acc: {:.2f}%  val_loss: {:.5f}  val_acc: {:.2f}%'
                          '   LMO: epochs={} loss= {:.4f}'
                          .format(t, self.args.n_modules, time.time() - time0, tr_loss, 100. * (1 - tr_error),
                                  vl_loss, 100. * (1 - vl_error), lmo_epochs, lmo_loss))
                    print('CURRENT BEST TRAIN RESULT:   Loss: {:.5f}   Accuracy: {:.2f}%   (Module {}, Epoch {})'
                          .format(self.best_train['train_loss'], 100. * (1. - self.best_train['train_error']),
                                  self.best_train['module'], self.best_train['epoch']))
                    print('CURRENT BEST VALDT RESULT:   Loss: {:.5f}   Accuracy: {:.2f}%   (Module {}, Epoch {})\n'
                          .format(self.best_validation['val_loss'], 100. * (1. - self.best_validation['val_error']),
                                  self.best_validation['module'], self.best_validation['epoch']))

            if early_stopping:
                if self.args.verbose >= 3:
                    if tr_loss < 1e-9:
                        print('EARLY STOPPING AT MODULE {} (training_loss={} is almost zero\n'.format(tr_loss))
                    elif t - self.best_train['module'] >= 10:
                        print('EARLY STOPPING AT MODULE {} (added {} modules with no improvement of '
                              'train error)\n'.format(t, t - self.best_train['module']))
                    else:
                        print('EARLY STOPPING AT MODULE {} (added {} modules with no improvement of '
                              'validation error)\n'.format(t, t - self.best_validation['module']))
                break

            time0 = time.time()

            if t < self.args.n_modules:
                # Frank Wolfe direction: use linear minimizing oracle LMO to find the best new Frank Wolfe direction
                name = 'mdl {}/{}'.format(t+1, self.args.n_modules)
                net = self.BasicModule(self.shape_in, self.dim_out, self.args.module_size, self.args.bound, self.afunc)
                lmo = LMO(net, train_buffer, self.args, name)
                train_log = lmo.train()
                epoch = lmo.best['epoch']
                lmo_loss = lmo.best['loss']
                if lmo_loss == float('inf'):
                    if self.args.verbose >= 3:
                        print('LMO loss is infinite. Probably NaN occurred due to high lr. Aborting training!')
                    break
                lmo_epochs = lmo.epoch
                s = self.freeze(lmo.best['net'])  # s is the FW direction
                # Away direction: find the worst direction among modules contained in sum_net
                v, idx, alpha = self.sum_net.find_away_direction(train_buffer)  # v is the away direction (a module)

                d = Direction(s, net2=v)  # Pairwise FW direction
                gamma = line_search(train_buffer, d, gamma_max=alpha, verbose=self.args.verbose)
                # gamma *= 0.8
                self.sum_net.append(net, gamma, idx)

                # print_mem(train_buffer, name='train_buffer')
                # print_mem(self.sum_net, name='sum_net')
                torch.cuda.empty_cache()
                # print('TENSORS MEMORY FOOTPRINT AFTER RELEASING BUFFERS')
                # print_gc_tensors()

                self.args.weight_decay *= factor  # reduce weight decay for the next module

        self.args.weight_decay = wd0  # return the original value of self.args.weight_decay
        return self.best_train, self.best_validation, log

    def eval(self, fw_buffer):
        eval_loss = 0.
        correct = n_eval = 0
        mae = mse = 0
        for x, y, z, z_grad, loss in fw_buffer:
            n_eval += x.shape[0]  # x.shape[0] = number of samples in this batch. Next, update loss incrementally
            eval_loss = ((n_eval - x.shape[0]) / n_eval) * eval_loss + (x.shape[0] / n_eval) * loss

            if self.args.regression:
                mae += F.l1_loss(z, y).item() * y.size(0)
                mse += F.mse_loss(z, y).item() * y.size(0)

            if not self.args.regression:  # classification task
                _, pred = z.max(1)  # get the index of the max probability
                target = y
                if len(y.shape) > 1:
                    _, target = y.max(1)
                correct += pred.eq(target).sum().item()
            elif self.args.dataset.endswith("_r"):  # classification posed as regression [-1, 1]
                pred = z.new_ones(z.shape, dtype=y.dtype)
                pred[z < 0] = -1
                correct += pred.eq(y).sum().item()

        mae /= n_eval
        mse /= n_eval
        rmse = mse ** 0.5
        error = 1.0 - correct / n_eval
        return eval_loss, error, rmse, mae

    def test(self, test_loader):
        assert self.best_validation['val_loss'] < float('inf'), "Model is not trained. Call train() first!"
        test_caching = 4 if self.args.caching else 0
        test_buffer = DataLoaderBufferFW(test_loader, test_caching, self.args.device, self.val_criterion,
                                         self.dim_out, transform_net=self.features_net, requires_grad=False)
        test_buffer.update(self.best_validation_model)
        return self.eval(test_buffer)

    @property
    def best_n_modules(self):  # number of modules/estimators in the final best model
        return self.best_validation['module']

    @staticmethod
    def inner_prod(fw_buffer, d):
        r"""return the inner product <grad, d(x, z)>"""
        inner_prod = 0.
        for x, _, z, z_grad, _ in fw_buffer:
            d_batch = d(x, z)
            d_batch *= z_grad
            inner_prod += d_batch.sum().item()
        return inner_prod

    @staticmethod
    def freeze(module):
        """freeze parameters of a module and put it to eval() mode, so it will not be trained by optimizer"""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module


def get_ws_param(args, attr):
    """get the corresponding warm start parameter, if it is not exists, use the value of the general parameter"""
    assert hasattr(args, attr), 'Invalid warm start parameter!'
    val = getattr(args, attr)
    if hasattr(args, 'ws_' + attr):
        ws_val = getattr(args, 'ws_' + attr)
        if isinstance(ws_val, str):
            ws_val = ws_val.strip()
        if ws_val or isinstance(ws_val, list) or isinstance(ws_val, int) or isinstance(ws_val, float):
            val = ws_val
    return val
