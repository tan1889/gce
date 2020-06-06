import argparse
from launcher import launch


parser = argparse.ArgumentParser(description='RANDOM FOREST ALGORITHM LAUNCHER')
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
                    help='P = auto | None | normalize | squash ...')
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

# Random forest optimization params
parser.add_argument('--criterion', type=str, default='auto', metavar='C',
                    help='criterion (loss) for training and eval (e.g. for eval validation set for early stopping). '
                         'None or auto means using mse for regression, gini for classification. '
                         'Alternative: mae for regression, entropy for classification.')  # mse=l2 mae=l1  entropy=nll
parser.add_argument('--n-estimators', type=int, default=1000, metavar='NE',
                    help='Number of estimators. If early-stopping > 0 -> maximum number of estimators')
parser.add_argument('--early-stopping', type=int, default=50, metavar='NE',
                    help='Number of early stopping rounds. 0 means disabled')
parser.add_argument('--min-n-estimators', type=int, default=10, metavar='NE',
                    help='(Only if early-stopping > 0) this specifies the minimum number of estimators to start with.')
parser.add_argument('--max-features', type=str, default='auto', metavar='MF',
                    help='MF = auto | sqrt | log2 | 0.1 | 0.2 | 1 | 2 ... '
                         'auto (sqrt, log2) means set to the (sqrt, log2) number of features.')
parser.add_argument('--max-depth', type=int, default=10, metavar='MD',
                    help='Often values 10 .. 150')
parser.add_argument('--min-samples-split', type=int, default=5, metavar='SS',
                    help='Often values 2 .. 15, if None then no limit')
parser.add_argument('--min-samples-leaf', type=int, default=5, metavar='SL',
                    help='Often values 1 .. 9')
parser.add_argument('--no-bootstrap', action='store_true', default=False,
                    help='Disable bootstrap of training set. Normally not required.')
parser.add_argument('--n-workers', type=int, default=-1, metavar='NW',
                    help='Number of workers for the algorithm. None means 1, -k ... means using n_cores-k cores. '
                         '-200 means using n_cores//2 cores, -300 means n_cores//3.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed, set to make the result reproducible. 0=disabled.')

# Other params
parser.add_argument('--save-result', type=int, default=1, metavar='SR',
                    help='SR = 0 | 1 | 2. 0: no saving. 1: save log file. 2: save log and best model.')
parser.add_argument('--verbose', type=int, default=3, metavar='V',
                    help='Level of verbose: 0 (silent), 1 (warning), 2 (info), '
                         '3 (model:summary), 4 (model:warning), 5 (model:details), 6 (model:debug).')

args = parser.parse_args()

args.no_cuda = True
args.algorithm = 'misc'
args.optimizer = 'rforest'

result = launch(args)
