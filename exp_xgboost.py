import argparse
from optimizers.modelselection import ModelSelection
from data_utils import ds_params
import torch.multiprocessing as mp


def neighbors(x, num, step=1, min_val=0., max_val=1e6):
    """return a list of n elements around x inclusive, whose value are in [min_val,max_val]"""
    neighbors = [x + i*step for i in range(-num//2, num//2) if min_val <= x + i*step <= max_val]
    if not neighbors:
        neighbors = [x]
    return neighbors


def args_search(args, large_ds=False, verbose=2, n_workers=-2, filename=None):
    # As there is early stopping, we just set n_estimators=2000 and don't need to tune it
    # First, we set large learning rate to tune other params (we will tune lr later)
    args.n_estimators = 2000
    args.early_stopping = 50
    args.eta = 0.1

    # Tune max_depth, min_child_weight first as they have highest impact
    # If the best value is at the end of the range, should extend the range
    print('\nTUNING max_depth, min_child_weight')
    args_grid = {'max_depth': [x for x in range(3, 14, 2)],
                 'min_child_weight': [x for x in range(0, 19, 2)]}

    # Gross grid search, 60 iters
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, filename=filename)
    res = ms.train()
    filename = ms.filename
    args = res['args']
    err = res['validation_score']
    print('Best params:  max_depth={}  min_child_weight={}'.format(args.max_depth, args.min_child_weight))

    # Fine grid search, 25 iters
    args_grid = {'max_depth': neighbors(args.max_depth, 5, 1, min_val=1),
                 'min_child_weight': neighbors(args.min_child_weight, 5, 1, min_val=0)}
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, filename=filename)
    res = ms.train()
    args = res['args']
    print('Best params:  max_depth={}  min_child_weight={}  error_diff: {:.4f})'.format(
        args.max_depth, args.min_child_weight, err - res['validation_score']))
    err = res['validation_score']

    print('\nTUNING gamma')
    args_grid = {'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, filename=filename)
    res = ms.train()
    args = res['args']
    print('Best params:  gamma={:.2f}  error_diff: {:.4f}'.format(
        args.gamma, err - res['validation_score']))
    err = res['validation_score']

    print('\nTUNING subsample, colsample_bytree')
    args_grid = {'subsample': [0.5, 0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9]}
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, filename=filename)
    res = ms.train()
    args = res['args']
    print('Best params:  subsample={:.2f}  colsample_bytree={:.2f}  err_diff={:.4f}'.format(
        args.subsample, args.colsample_bytree, err - res['validation_score']))
    err = res['validation_score']

    print('\nTUNING regularization parameters lambda, alpha')
    args_grid = {'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                 'reg_alpha': [0, 0.1, 1]}
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, filename=filename)
    res = ms.train()
    args = res['args']
    print('Best params:  lambda={:.2f}  alpha={:.2f}  err_diff={:.4f}'.format(
        args.reg_lambda, args.reg_alpha, err - res['validation_score']))
    err = res['validation_score']

    print('\nTUNING learning rate eta')
    args_grid = {'eta': [0.2, 0.15, 0.05, 0.01, 0.005, 0.001]}
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, filename=filename)
    res = ms.train()
    args = res['args']
    print('Best params:  learning rate eta={:.4f}  err_diff={:.4f}'.format(
        args.eta, err - res['validation_score']))
    err = res['validation_score']

    print('\nFINAL TUNING random search for all params')  # above used 146 iters, here we do additional random search
    args_grid = {'eta': neighbors(args.eta, 6, step=args.eta / 5, min_val=0.00001, max_val=0.2),
                 'max_depth': neighbors(args.max_depth, 6, step=1, min_val=2, max_val=16),
                 'min_child_weight': neighbors(args.min_child_weight, 6, step=1, min_val=0, max_val=20),
                 'gamma': neighbors(args.gamma, 6, step=0.05, min_val=0, max_val=0.5),
                 'subsample': neighbors(args.subsample, 6, step=0.05, min_val=0.45, max_val=0.95),
                 'colsample_bytree': neighbors(args.colsample_bytree, 6, step=0.05, min_val=0.45, max_val=0.95),
                 'reg_lambda': neighbors(args.reg_lambda, 6, step=args.reg_lambda / 6., min_val=0),
                 'reg_alpha': neighbors(args.reg_alpha, 6, step=args.reg_alpha / 6., min_val=0)
                 }
    n_iter = 100 if large_ds else 2000
    ms = ModelSelection(args, args_grid, verbose=verbose, n_workers=n_workers, n_iter=n_iter, filename=filename)
    res = ms.train()
    args = res['args']
    print('err_diff={:.4f}   BEST ARGS: {}'.format(err - res['validation_score'], args))


# ======================================== #
# ============= MAIN ===================== #
# ======================================== #

args = argparse.Namespace(desc='exp_xgb', dataset='', preprocessing='default',
                          validation_ratio=0.2, expansion=0, no_shuffle=False,
                          batch_size=256, test_batch_size=256, n_workers_dl=1,
                          algorithm='misc', optimizer='xgboost',
                          criterion='auto', n_estimators=2000,
                          early_stopping=50, max_depth=5,
                          eta=0.1, min_child_weight=1, gamma=0.,
                          subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=1., reg_alpha=0., scale_pos_weight=1.,
                          n_workers=1, seed=1, no_cuda=False,
                          save_result=0, verbose=5
                          )
n_cpu = mp.cpu_count()

for args.dataset in ['diabetes', 'boston', 'housing', 'msd',
                     'iris', 'wine', 'breast_cancer', 'digits',
                     'cifar10_f', 'mnist', 'covertype', 'kddcup99']:

    args.validation_ratio = 0.25 if args.dataset == 'covertype' else 0.2
    ds = ds_params[args.dataset]

    args.criterion = 'rmse' if ds['y_dim'] == 1 else 'error'

    f = 'output/{}_{}_{}_{}_tuning.pkl'.format(
        args.desc, args.dataset, args.optimizer, args.criterion)

    args.n_workers = n_cpu
    large_ds = ds['n_train'] >= 10000
    args_search(args, large_ds=large_ds, verbose=2, n_workers=0, filename=f)
