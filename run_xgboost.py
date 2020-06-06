import argparse
from launcher import launch


parser = argparse.ArgumentParser(description='XGBOOST ALGORITHM LAUNCHER')
parser.add_argument('--desc', type=str, default='', metavar='D',
                    help='Experiment description, prefix of the output filename, good for grouping output files.')

# Dataset params
parser.add_argument('--dataset', type=str, default='iris', metavar='DS',
                    help='DS = mnist | mnist_t | cifar10 | cifar10_t | cifar10_f '
                         '| covertype | kddcup99 | susy | susy_r | pufs | pufs_r | msd | housing | transcode '
                         '| boston | iris | diabetes | digits | wine | breast_cancer.\n'
                         'postfixes: '
                         '"_r": posing a two class classification problem as a regression [-1, 1] problem, '
                         '"_t": transformation is applied (elastic for MNIST and random crop/flip for CIFAR10), '
                         '"_f": (only for cifar10) cifar10_t go through features_net cifar10_342f.pt. '
                         'Note: Datasets are from UCI ML Repository and sklearn')
parser.add_argument('--preprocessing', type=str, default=None, metavar='P',
                    help='P {None=auto, normalize, squash ...}')
parser.add_argument('--validation-ratio', type=float, default=0.2, metavar='R',
                    help='Use R * n_train samples from train_data for validation. '
                         'if R = 0 (default): use test_data for validation (to see improvement in test performance)')
parser.add_argument('--expansion', type=int, default=0, metavar='E',
                    help='This option only works for MNIST and CIFAR10 '
                         'If E >= 2, expand the dataset using the appropriate transformation (new_size=E*old_size). '
                         'Note: if transformation is enabled (by using dataset name + _t), and E <= 1, the '
                         'transformation is on-the-fly, so each request to the same data item will be returned with'
                         'a different (transformed) item. If E >= 2, the transformation is offline, the dataset is'
                         'E times larger and static, with the first n items identical to original dataset, and the '
                         'next items are their transformation in the same ordering. However, if E>MAX_CACHEABLE (10) '
                         'we simulate online transformation using Ex offline transformations: dataset_size = n, but '
                         'query to image i is returned with one of its E transformed instances.')
parser.add_argument('--batch-size', type=int, default=128, metavar='B',
                    help='Batch size for training (default: B = 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='BT',
                    help='Batch size for testing (default: BT = 128)')
parser.add_argument('--n-workers-dl', type=int, default=1, metavar='NW',
                    help='Number of workers for dataloader. None means 0: run in the same process. '
                         '5 means using min(n_cores, 5). -5 means using max(1, n_cores-5). '
                         '-200 means using n_cores//2. -300 means using n_cores//3!')
parser.add_argument('--no-shuffle', action='store_true', default=False,
                    help='Disable shuffling of training set. Normally not required.')

# XGBoost params. 'objective' is automatically set depending on the dataset (regression or classification
parser.add_argument('--criterion', type=str, default='auto', metavar='C',
                    help='The eval_metric param: Eval criterion (e.g. for eval validation set for early stopping). '
                         'None or auto means using rmse for regression, error for 2-class, merror for multi-class. '
                         'Alternative: mae for regression, logloss for multi-class, auc for 2-class.')
parser.add_argument('--n-estimators', type=int, default=1000, metavar='NE',
                    help='The num_rounds parameter: Number of rounds (each round adds one estimator): '
                         'Typical value: 10 - 2000')
parser.add_argument('--early-stopping', type=int, default=20, metavar='NE',
                    help='Number of early stopping rounds. 0 means disabled')
parser.add_argument('--max-depth', type=int, default=5, metavar='MD',
                    help='The maximum depth of a tree. Used to control over-fitting as higher depth will allow the '
                         'model to learn relations very specific to a particular sample.. Typical value: 3 - 10')
parser.add_argument('--eta', type=float, default=0.1, metavar='LR',
                    help='eta is learning rate. Typical value: 0.01 - 0.2')
parser.add_argument('--min-child-weight', type=int, default=1, metavar='CW',
                    help='Defines the minimum sum of weights of all observations required in a child. Used to control '
                         'over-fitting. Higher values prevent a model from learning relations which might be highly '
                         'specific to the particular sample selected for a tree. Typical value: 0 - 1e3')
parser.add_argument('--gamma', type=float, default=0.1, metavar='LR',
                    help='Gamma specifies the minimum loss reduction required to make a split. Typical value: 0 - 0.5')
parser.add_argument('--subsample', type=float, default=0.8, metavar='LR',
                    help='The fraction of observations to be randomly sampled for each tree. Lower values make the '
                         'algorithm more conservative and prevents overfitting but too small values might lead to '
                         'under-fitting. Typical value: 0.5 - 1')
parser.add_argument('--colsample-bytree', type=float, default=0.8, metavar='LR',
                    help='The fraction of columns to be randomly sampled for each tree. Typical value: 0.5 - 1')
parser.add_argument('--reg-lambda', type=float, default=1., metavar='LR',
                    help='L2 regularization term on weights. Commonly, gamma is used for regularization, but this '
                         'can be used as well. Typical value: 0 - 1e2.')
parser.add_argument('--reg-alpha', type=float, default=0., metavar='LR',
                    help='L1 regularization term on weights. Commonly, gamma is used for regularization, but this '
                         'can be used as well, specially for high dimension problem. Typical value: 0 - 1e2.')
parser.add_argument('--scale-pos-weight', type=float, default=1., metavar='LR',
                    help='Control the balance of positive and negative weights, useful for unbalanced classes. '
                         'A typical value to consider: sum(negative instances) / sum(positive instances)')

# Other params
parser.add_argument('--n-workers', type=int, default=4, metavar='NW',
                    help='Number of workers for the algorithm. None means 1, -k ... means using n_cores-k cores. '
                         '-200 means using n_cores//2 cores, -300 means n_cores//3.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed, set to make the result reproducible. 0=disabled.')
parser.add_argument('--save-result', type=int, default=1, metavar='SR',
                    help='SR = 0 | 1 | 2. 0: no saving. 1: save log file. 2: save log and best model.')
parser.add_argument('--verbose', type=int, default=5, metavar='V',
                    help='Level of verbose: 0 (silent), 1 (warning), 2 (info), '
                         '3 (model:summary), 4 (model:warning), 5 (model:details), 6 (model:debug).')

args = parser.parse_args()

args.algorithm = 'misc'
args.optimizer = 'xgboost'

result = launch(args)
