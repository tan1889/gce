import importlib
import time
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import pickle
from data_utils import *
import os.path
import errno
from config import *
import random
import numpy
import multiprocessing
import warnings
import os
import time


def launch(args, ds=None):

    if args.verbose == 0:  # suppress warnings from data_utils
        warnings.simplefilter("ignore")

    t0 = time.time()

    set_auto_params(args)

    # -------------- load dataset and check common parameters -------------- #

    args.algorithm = args.algorithm.lower()

    inform(args, 'Loading {} dataset...'.format(args.dataset))
    if ds:
        train_loader, val_loader, test_loader, x_shape, out_dim, args.regression = ds
    else:
        train_loader, val_loader, test_loader, x_shape, out_dim, args.regression = load_dataset(args)
    inform(args, '   n_train={} n_val={} n_test={} x_shape={} out_dim={}  [n_workers_dl={}]'.format(
        len(train_loader), len(val_loader), len(test_loader), x_shape, out_dim, args.n_workers_dl))

    set_random_seed(args)

    if args.desc:
        inform(args, '\nRUN DESCRIPTION:   {}   --> prepend to output filename'.format(args.desc))

    # -------------- load and train the model ------------------------------ #

    time_start = datetime.now()

    if args.algorithm == 'misc':
        model, desc = load_misc_model(train_loader, val_loader, x_shape, out_dim, args)
    elif args.algorithm.endswith('_cnn'):
        model, desc = load_cnn_model(train_loader, val_loader, x_shape, out_dim, args)
    else:
        model, desc = load_nn_model(train_loader, val_loader, x_shape, out_dim, args)

    best_train, best_val, train_log = model.train()
    test_loss, test_error, test_rmse, test_mae = model.test(test_loader)

    # -------------- print and save the result ----------------------------- #

    if args.regression and not args.dataset.endswith("_r"):
        inform(args, 'BEST RESULT ACHIEVED AT MODULE {} EPOCH {}'
               '\n  - TRAIN:        loss={:.4f}   rmse={:.4f}   mae={:.4f}'
               '\n  - VALIDATION:   loss={:.4f}   rmse={:.4f}   mae={:.4f}'
               '\n  - TEST:         loss={:.4f}   rmse={:.4f}   mae={:.4f}'
               .format(best_val['module'], best_val['epoch'],
                       best_val['train_loss'], best_val['train_rmse'], best_val['train_mae'],
                       best_val['val_loss'], best_val['val_rmse'], best_val['val_mae'],
                       test_loss, test_rmse, test_mae))
    else:
        inform(args, 'BEST RESULT ACHIEVED AT MODULE {} EPOCH {}'
               '\n  - TRAIN:        loss={:.5f}   error={:.2f}%   accuracy={:.2f}%'
               '\n  - VALIDATION:   loss={:.5f}   error={:.2f}%   accuracy={:.2f}%'
               '\n  - TEST:         loss={:.5f}   error={:.2f}%   accuracy={:.2f}%'
               .format(best_val['module'], best_val['epoch'],
                       best_val['train_loss'], 100 * best_val['train_error'], 100 - 100 * best_val['train_error'],
                       best_val['val_loss'], 100 * best_val['val_error'], 100 - 100 * best_val['val_error'],
                       test_loss, 100 * test_error, 100 - 100 * test_error))

    run_time = time.time() - t0
    inform(args, '\nRUN_TIME={:.2f}s'.format(run_time))

    if args.regression:
        val_score = best_val['val_loss']
    else:
        val_score = best_val['val_error']

    result = {'test': {'loss': test_loss, 'error': test_error, 'rmse': test_rmse, 'mae': test_mae},
              'best_validation': best_val, 'best_train': best_train,
              'validation_score': val_score, 'best_n_modules': model.best_n_modules,
              'run_time': run_time, 'filename': None, 'args': args, 'modules_log': train_log}

    f = None
    if args.save_result > 0:
        f = 'output/{}{}_{:%Y%m%d.%H%M}_{:%H%M}_{}_'.format(
            args.desc + '_' if args.desc else '', args.dataset, time_start, datetime.now(), desc)

        if args.algorithm == 'misc':
            train_score = best_train['train_loss'] if args.regression else 100. - 100 * best_train['train_error']
            test_score = test_loss if args.regression else 100. - 100 * test_error
            f2 = 'train_{:.4f}_test_{:.4f}.pkl'.format(train_score, test_score)
            pickle.dump(result, open(f + f2, "wb"))  # save log file
            inform(args, '\nLOG FILE:   {}'.format(f))
            if args.save_result > 1:  # save model
                f2 = f2.replace('.pkl', '_best_model.' + args.optimizer)
                pickle.dump(model.best_validation_model, open(f + f2), "wb")
                inform(args, 'BEST MODEL FILE:' + f)
        else:
            if args.regression and not args.dataset.endswith("_r"):
                f2 = 'train_{:.4f}_test_{:.4f}.pkl'.format(best_train['train_loss'], test_loss)
            else:
                f2 = 'train_{:.4f}_{:.2f}_test_{:.4f}_{:.2f}.pkl'.format(
                        best_train['train_loss'], 100. * (1. - best_train['train_error']),
                        test_loss, 100. * (1. - test_error))

            pickle.dump(result, open(f + f2, "wb"))
            inform(args, '\nLOG FILE:   {}'.format(f))

            if args.save_result > 1:  # save model
                state_dict = model.best_validation_model.cpu().state_dict()
                fm = f + f2.replace('.pkl', '_best_val_state_dict.pt')
                torch.save(state_dict, fm)
                inform(args, 'BEST VAL MODEL FILE:   {}_best_val_state_dict.pt'.format(fm))

                state_dict = model.best_train_model.cpu().state_dict()
                fm = f + f2.replace('.pkl', '_best_train_state_dict.pt')
                torch.save(state_dict, fm)
                inform(args, 'BEST TRAIN MODEL FILE:   {}_best_train_state_dict.pt'.format(fm))

    result['filename'] = f
    return result


def set_auto_params(args):  # set params that require automatic setting

    if args.algorithm in {'pfw', 'fw', 'afw'} or (hasattr(args, 'model') and args.model.startswith('CN')):
        if hasattr(args, 'bound') and args.bound <= 0:
            assert args.dataset in ds_params, 'Can not set bound automatically. Dataset info not available!'
            if ds_params[args.dataset]['y_dim'] > 1:  # classification
                args.bound = 10
            else:
                args.bound = 4 * ds_params[args.dataset]['y_bound'] / 3

        if args.batch_size <= 0:
            assert args.dataset in ds_params, 'Can not set batch_size automatically. Dataset info not available!'
            args.batch_size = min(10000, 1 + ds_params[args.dataset]['n_train'] // 3)

        if args.test_batch_size <= 0:
            args.test_batch_size = 256

    # covertype -> fixed validation ratio 0.25 per original dataset specs
    if args.dataset == 'covertype' and args.validation_ratio > 0:
        args.validation_ratio = 0.25


def load_dataset(args):
    make_dirs()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    args.dataset = args.dataset.lower()

    n_cpu = multiprocessing.cpu_count()
    if not hasattr(args, 'n_workers_dl') or args.n_workers_dl is None:
        args.n_workers_dl = 1
        warnings.warn('n_workers_dl=None -> set to 1')
    elif args.n_workers_dl > n_cpu:
        args.n_workers_dl = n_cpu
        warnings.warn('n_workers_dl is too large, set to num_cores=' + str(n_cpu))
    elif args.n_workers_dl <= -300:
        args.n_workers_dl = max(1, n_cpu // 3)
        warnings.warn('n_workers_dl <= -300 -> set to num_cores//3={}'.format(args.n_workers_dl))
    elif args.n_workers_dl <= -200:
        args.n_workers_dl = max(1, n_cpu // 2)
        warnings.warn('n_workers_dl <= -200 -> set to num_cores//2={}'.format(args.n_workers_dl))
    elif args.n_workers_dl < -0:
        args.n_workers_dl = max(1, n_cpu + args.n_workers_dl)
        warnings.warn('n_workers_dl = -k -> set to num_cores - k={}'.format(args.n_workers_dl))

    transform = args.dataset.endswith('_t')
    exp = getattr(args, 'expansion', 0)

    if args.algorithm.endswith('_cnn'):
        transform = not args.no_transform
        args.preprocessing = 'transform, normalize' if transform else 'normalize'
        return load_cnn_dataset(args.dataset.replace('_t', ''),
                                args.cuda, args.n_workers_dl, not args.no_shuffle,
                                validation_ratio=args.validation_ratio, transform=transform,
                                train_batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    if args.dataset.startswith('mnist'):
        args.preprocessing = 'transform, normalize' if transform else 'normalize'
        return load_mnist(args.cuda, args.n_workers_dl, validation_ratio=args.validation_ratio,
                          transform=transform, expansion=exp,
                          train_batch_size=args.batch_size, test_batch_size=args.test_batch_size, verbose=args.verbose)

    elif args.dataset.startswith('cfr10') or args.dataset.startswith('cifar10'):
        args.preprocessing = 'transform, normalize' if transform else 'normalize'
        if args.dataset.endswith('_f'):
            return load_cifar10f(args.cuda, args.n_workers_dl,
                                 validation_ratio=args.validation_ratio, expansion=exp,
                                 train_batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                                 verbose=args.verbose)
        else:
            return load_cifar10(args.cuda, args.n_workers_dl, validation_ratio=args.validation_ratio,
                                transform=transform, expansion=exp,
                                train_batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                                verbose=args.verbose)

    if not args.preprocessing or 'auto' in args.preprocessing or 'default' in args.preprocessing:
        args.preprocessing = None

    if args.dataset == 'covertype':
        if args.validation_ratio > 0. and args.validation_ratio != 0.25:
            warnings.warn('val_ratio={:.2f} is invalid for CoverType -> forcing 0.25 per dataset specs'.format(
                args.validation_ratio))
            args.validation_ratio = 0.25  # CoverType has 25% validation set
    elif args.dataset.endswith('_r'):
        warnings.warn('overriding preprocessing=normalize,regression for 2-class posing as regression task')
        args.preprocessing = 'normalize, regression'

    func_name = 'load_' + args.dataset.replace('_r', '')
    if func_name in globals():
        load_func = globals()[func_name]
        if args.preprocessing:
            return load_func(args.cuda, args.n_workers_dl, validation_ratio=args.validation_ratio,
                             preprocessing=args.preprocessing,
                             train_batch_size=args.batch_size, test_batch_size=args.test_batch_size)
        else:  # use the default preprocessing option
            return load_func(args.cuda, args.n_workers_dl, validation_ratio=args.validation_ratio,
                             train_batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    else:
        func_name = 'load_toy_dataset'
        load_func = globals()[func_name]
        if args.preprocessing:
            return load_func(args.dataset, args.cuda, args.n_workers_dl, validation_ratio=args.validation_ratio,
                             preprocessing=args.preprocessing,
                             train_batch_size=args.batch_size, test_batch_size=args.test_batch_size)
        else:  # use the default preprocessing option
            return load_func(args.dataset, args.cuda, args.n_workers_dl, validation_ratio=args.validation_ratio,
                             train_batch_size=args.batch_size, test_batch_size=args.test_batch_size)


def load_misc_model(train_loader, val_loader, in_shape, out_dim, args):
    import models.misc
    model = models.misc.Model(train_loader, val_loader, in_shape, out_dim, args)

    desc = '{}_n{}_{}'.format(args.optimizer, args.n_estimators, args.criterion)
    if args.optimizer == 'xgboost':
        desc += '_lr{}_dp{}_cw{}_gm{}_ss{}_cb{}_ld{}'.format(
            args.eta, args.max_depth, args.min_child_weight, args.gamma,
            args.subsample, args.colsample_bytree, args.reg_lambda)
    elif args.optimizer == 'rforest':
        desc += '_ft.{}_sl{}_dp{}_ss{}'.format(
            args.max_features, args.min_samples_leaf, args.max_depth, args.min_samples_split)
    return model, desc


def load_nn_model(train_loader, val_loader, in_shape, out_dim, args):
    if args.caching > 0 and not train_loader.cacheable:
        args.caching = 0
        warnings.warn('caching=1 but dataset is not cacheable -> forcing caching=0!')

    inform(args, '\n{:15}\tdataset={}   val_ratio={}   model={}.{}\n'
                 '\t\t\t\tdevice={}   loss={}   activation={}   bound={}'.format(
                    'REGRESSION:' if args.regression else 'CLASSIFICATION:',
                    args.dataset, args.validation_ratio, args.algorithm, args.model,
                    args.device, args.loss, args.activation, args.bound))

    greedy_str = ''
    if args.algorithm in {'pfw', 'glrn', 'dgn', 'wgn', 'afw', 'fw'}:
        ws = ws2 = '0'
        if args.warm_start:
            ws = '1.lr{}d{}'.format(args.ws_lr, args.ws_weight_decay)
            ws2 = '1 ({} lr={} d={})'.format(args.ws_optimizer, args.ws_lr, args.ws_weight_decay)
        greedy_str = 'x{}_ws{}'.format(args.n_modules, ws)
        inform(args, 'GREEDY ALGO:\tn_modules={}   module_size={}   warm_start={}'.format(
            args.n_modules, args.module_size, ws2))

    bound_str = ''
    if args.model.startswith('CN') or args.model.startswith('Convex') \
            or args.algorithm.endswith('ern') or greedy_str:
        bound_str = '_B{}'.format(args.bound)

    inform(args, 'OPTIMIZER:\t\t{}   n_epochs={}   lr={}   momentum={}   decay={}   decay_bias={}'.format(
        args.optimizer, args.n_epochs, args.lr, args.momentum, args.weight_decay, not args.no_bias_decay))

    if args.milestones:
        if isinstance(args.milestones[0], list):
            lr_scheduler = 'cst{}'.format(
                str(args.milestones).replace('], [', '_').replace('[', '').replace(']', '').replace(', ', '-'),
            )
            inform(args, 'LR_SCHEDULER:\tcustom lr, milestones={}, lrs={}, n_epochs={}'.format(
                args.milestones[0], args.milestones[1], args.n_epochs))
        else:
            lr_scheduler = 'mlt{}_f{}_min{}'.format(
                str(args.milestones).replace('[', '').replace(']', '').replace(', ', '-'), args.factor, args.min_lr
            )
            inform(args, 'LR_SCHEDULER:\tmulti-step decay   gamma={}   milestones={}   min_lr={}\n'.format(
                args.factor, 'auto' if args.milestones == [0] else args.milestones, args.min_lr))
    elif args.patience <= 0:
        lr_scheduler = 'exp_f{}_min{}'.format(args.factor, args.min_lr)
        inform(args, 'LR_SCHEDULER:\texponential decay each epoch   gamma={}   min_lr={}\n'.format(
            args.factor, args.min_lr))
    else:
        lr_scheduler = 'ptn{}_f{}_min{}_th{}'.format(args.patience, args.factor, args.min_lr, args.threshold)
        inform(args, 'LR_SCHEDULER:\tdecay on plateau   gamma={}   '
                     'patience={}   min_lr={}   threshold={}\n'.format(
                         args.factor, args.patience, args.min_lr, args.threshold))

    features_net = None
    if args.features_net:
        f = os.path.join('features_nets/', args.features_net)
        assert os.path.exists(f), 'features_nets/{} does not exist'.format(args.features_net)
    
        if 'mnist' in args.features_net:
            inform(args, 'Loading MNIST features extractor:' + f)
            assert 'mnist' in args.dataset, 'Can not use mnist features_net for dataset ' + args.dataset
            import models.features
            if '_t.' in args.features_net:
                features_net = models.features.MnistTFeatures()
                in_shape = (3, 3, 128)
            else:
                features_net = models.features.MnistFeatures()
                in_shape = (10, 10, 32)
        elif 'cifar10' in args.features_net:
            inform(args, 'Loading CIFAR10 features extractor: ' + f)
            assert 'cifar10' in args.dataset, 'Can not use cifar10 features_net for dataset ' + args.dataset
            import models.features
            if '_342f' in args.features_net:
                in_shape = (342, 1, 1)
                features_net = models.features.Cifar10F_342x1x1()
            else:
                in_shape = (342, 4, 4)
                features_net = models.features.Cifar10F_342x4x4()
        else:
            raise Exception('Unknown features net: ' + args.features_net)

        features_net.load_state_dict(torch.load(f))
        for param in features_net.parameters():
            param.requires_grad = False
        features_net.eval()
        features_net.to(args.device)

    criterion = get_criterion(args.loss, args.regression)
    val_criterion = criterion
    if args.validation_loss and args.validation_loss != args.loss and args.validation_loss not in {'auto', 'default'}:
        val_criterion = get_criterion(args.validation_loss, args.regression)

    if args.activation in 'relu':
        afunc = F.relu
    elif args.activation == 'sigmoid':
        afunc = F.sigmoid
    elif args.activation == 'tanh':
        afunc = torch.tanh
    elif args.activation == 'softsign':
        afunc = F.softsign
    else:
        raise Exception('Unknown activation function: ' + args.activation)

    try:
        alg = importlib.import_module('models.' + args.algorithm)
    except ImportError:
        raise
    model = alg.Model(train_loader, val_loader, criterion, val_criterion, afunc, in_shape, out_dim, args, features_net)

    if args.algorithm.endswith('ern'):
        lrn_str = ''
        for n, w, o in args.res_sizes:
            lrn_str += '{}x{}.{}_'.format(n, w, o)
        if args.algorithm == 'ern':
            model_desc = '{}_{}{}'.format(args.algorithm, lrn_str, args.model)
        else:
            model_desc = '{}_{}{}.{}'.format(args.algorithm, lrn_str, args.model, args.module_size)
    else:
        model_desc = '{}_{}.{}'.format(args.algorithm, args.model, args.module_size)

    desc = '{}{}{}_{}_{}_{}_e{}_lr{}_d{}{}_{}'.format(
        model_desc, greedy_str, bound_str, args.loss, args.activation, args.optimizer,
        args.n_epochs, args.lr, args.weight_decay, 'b0' if args.no_bias_decay else 'b1', lr_scheduler)

    return model, desc


def load_cnn_model(train_loader, val_loader, input_shape, num_classes, args):
    inform(args, '\n{:15}\tdataset={}  val_ratio={}  model={}.{}  device={}  bound={}'.format(
        'REGRESSION:' if args.regression else 'CLASSIFICATION:',
        args.dataset, args.validation_ratio, args.algorithm, args.model,
        args.device, args.bound))

    inform(args, 'OPTIMIZER:\t\t{}  n_epochs={}  lr={}  momentum={}  decay={}  decay_bias=True'.format(
        args.optimizer, args.n_epochs, args.lr, args.momentum, args.weight_decay))

    if args.milestones:
        if isinstance(args.milestones[0], list):
            lr_scheduler = 'cst{}'.format(
                str(args.milestones).replace('], [', '_').replace('[', '').replace(']', '').replace(', ', '-'),
            )
            inform(args, 'LR_SCHEDULER:\tcustom lr, milestones={}, lrs={}, n_epochs={}'.format(
                args.milestones[0], args.milestones[1], args.n_epochs))
        else:
            lr_scheduler = 'mlt{}_f{}_min{}'.format(
                str(args.milestones).replace('[', '').replace(']', '').replace(', ', '-'), args.factor, args.min_lr
            )
            inform(args, 'LR_SCHEDULER:\tmulti-step decay  gamma={}  milestones={}  min_lr={}\n'.format(
                args.factor, 'auto' if args.milestones == [0] else args.milestones, args.min_lr))
    elif args.patience <= 0:
        lr_scheduler = 'exp_f{}_min{}'.format(args.factor, args.min_lr)
        inform(args, 'LR_SCHEDULER:\texponential decay each epoch  gamma={}  min_lr={}\n'.format(
            args.factor, args.min_lr))
    else:
        lr_scheduler = 'ptn{}_f{}_min{}_th{}{}'.format(args.patience, args.factor, args.min_lr,
                                                       args.threshold, args.threshold_mode)
        inform(args, 'LR_SCHEDULER:\tdecay on plateau  gamma={}  patience={}  min_lr={}  threshold={}{}\n'.format(
            args.factor, args.patience, args.min_lr, args.threshold, args.threshold_mode))
    try:
        alg = importlib.import_module('models.' + args.algorithm)
    except ImportError:
        raise
    model = alg.Model(train_loader, val_loader, input_shape, num_classes, args)

    desc = '{}_B{}_{}_e{}_lr{}_d{}b1_{}'.format(
        args.model, args.bound, args.optimizer, args.n_epochs,
        args.lr, args.weight_decay, lr_scheduler)
    return model, desc


def get_criterion(loss, regression):
    loss = loss.lower()
    if loss in {'auto', 'default'}:
        if regression:
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    elif loss in {'nll', 'cross_entropy', 'entropy'}:
        assert not regression, 'Can not use NLL loss for regression task!'
        return nn.CrossEntropyLoss()
    elif loss in {'l2', 'mse'}:
        return nn.MSELoss()
    elif loss in {'l1', 'mae'}:
        return nn.L1Loss()
    else:
        raise Exception('Unknown loss type: ' + loss)


def make_dirs():
    # create the datasets and checkpoints directories if not exists
    try:
        os.makedirs(config['DEFAULT']['datasets'])
        os.makedirs(config['DEFAULT']['checkpoints'])
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def set_random_seed(args):
    seed = args.seed
    if args.seed > 0:
        inform(args, 'args.seed={}, setting random seed to {}'.format(args.seed, seed))
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
    return args.seed


def inform(args, s):
    if args.verbose >= 2:
        print(s)


def generate_summary_list(files):
    """from a list of files generate appropriate list of modules_log (discard epochs_log)"""
    results = []
    for f in files:
        result = pickle.load(open(f, 'rb'))
        result['filename'] = f
        for m in result['modules_log']:
            m['log'] = []
        results.append(result)
    return results
