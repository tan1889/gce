import argparse
from optimizers.modelselection import ModelSelection
from data_utils import ds_params


args = argparse.Namespace(desc='exp_gce', dataset='', preprocessing='default',
                          validation_ratio=0.2, expansion=0, no_shuffle=False,
                          n_workers_dl=1, features_net=None,
                          algorithm='pfw', model='SN2h',
                          activation='relu', bound=10,
                          module_size=10, n_modules=100,
                          early_stopping=10, min_n_modules=1,
                          filename=None,
                          loss='', validation_loss='',
                          batch_size=256, test_batch_size=256,
                          optimizer='adam', n_epochs=20000,
                          lr=0.001, momentum=0.9,
                          weight_decay=0., min_weight_decay=0., no_bias_decay=False,
                          factor=0.1, patience=10,
                          threshold=1e-1, threshold_mode='adp',
                          min_lr=1e-5, milestones=[],
                          warm_start=0,
                          caching=2, seed=1, no_cuda=False,
                          save_result=2, verbose=4
                          )

bsizes = {'diabetes': 94, 'boston': 108, 'housing': 4403, 'msd': 10000,
          'iris': 32, 'wine': 38, 'breast_cancer': 122, 'digits': 384,
          'cifar10_f': 10000, 'mnist': 10000, 'covertype': 3780, 'kddcup99': 10000}


for args.dataset in bsizes.keys():
    args.batch_size = bsizes[args.dataset]
    args.validation_ratio = 0.25 if args.dataset == 'covertype' else 0.2
    ds = ds_params[args.dataset]

    args.module_size = 10 if ds['n_train'] < 10000 else 100

    # default module_size, bound, loss for classification & large datasets
    args_grid = {'module_size': [100], 'bound': [10]}
    args.loss = args.validation_loss = 'nll'

    if ds['y_dim'] == 1:  # regression -> set bound and loss correspondingly
        args.loss, args.validation_loss = 'mse', 'mae'
        args_grid['bound'] = [round(3*ds['y_bound']), round(2*ds['y_bound']), round((4/3)*ds['y_bound'])]

    if ds['n_train'] < 10000:  # small datasets -> set module_size correspondingly
        args_grid['module_size'] = [1, 10]

    f = 'output/{}_{}_{}_{}_tuning.pkl'.format(
            args.desc, args.dataset, args.algorithm, args.validation_loss)

    ms = ModelSelection(args, args_grid, verbose=3, n_workers=0, filename=f)
    res = ms.train()

    print('\nBEST ARGS:', res['args'], '\n')
