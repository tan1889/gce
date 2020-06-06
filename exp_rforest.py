import argparse
from optimizers.modelselection import ModelSelection
from data_utils import ds_params
import torch.multiprocessing as mp


args = argparse.Namespace(desc='exp_rf', dataset='', preprocessing='default',
                          validation_ratio=0.2, expansion=0, no_shuffle=False,
                          batch_size=256, test_batch_size=256, n_workers_dl=1,
                          algorithm='misc', optimizer='rforest', criterion='mse',
                          n_estimators=2000, early_stopping=50,
                          min_n_estimators=10, step_size=10,
                          max_features='auto', max_depth=10,
                          min_samples_split=5, min_samples_leaf=5,
                          no_bootstrap=False, n_workers=1, seed=1,
                          save_result=0, no_cuda=False, verbose=3
                          )
n_cpu = mp.cpu_count()

for args.dataset in ['diabetes', 'boston', 'housing', 'msd',
                     'iris', 'wine', 'breast_cancer', 'digits',
                     'cifar10_f', 'mnist', 'covertype', 'kddcup99']:

    args.validation_ratio = 0.25 if args.dataset == 'covertype' else 0.2
    ds = ds_params[args.dataset]

    args.criterion = 'mse' if ds['y_dim'] == 1 else 'gini'

    f = 'output/{}_{}_{}_{}_tuning.pkl'.format(
        args.desc, args.dataset, args.optimizer, args.criterion)

    args_grid = {'max_features': ['auto', 'sqrt', 'log2', 1, 3, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                 'min_samples_leaf': [1, 2, 5, 10, 20, 30, 50, 80, 120, 170, 230],
                 'max_depth': [1, 2, 5, 10, 20, 30, 50, 80, 120],
                 'min_samples_split': [x for x in range(2, 17, 2)],
                 'no_bootstrap': [True, False]  # usually False is recommended
                 }
    args.n_workers = n_cpu
    n_iters = 2000 if ds['n_train'] < 10000 else 400
    ms = ModelSelection(args, args_grid, verbose=2, n_iter=n_iters, n_workers=0, filename=f)

    res = ms.train()

    print('\nBEST ARGS:', res['args'], '\n')
