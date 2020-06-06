"""launch a nn model from command line with all parameters and their description.
nn models are: classical nn, greedy variants (pfw, afw, fw, wgn), Lipschitz constrained ResNet (lrn, glrn)"""
import argparse
from launcher import launch


parser = argparse.ArgumentParser(description='ALGORITHMS LAUNCHER')
parser.add_argument('--desc', type=str, default='', metavar='D',
                    help='Experiment description, goes to prefix of output filename, good for grouping output files.')

# Dataset params
parser.add_argument('--dataset', type=str, default='digits', metavar='D',
                    help='D = mnist | mnist_t | cifar10 | cifar10_t | cifar10_f '
                         '| covertype | kddcup99 | susy | susy_r | pufs | pufs_r | msd | housing | transcode '
                         '| boston | iris | diabetes | digits | wine | breast_cancer.\n'
                         'postfixes: '
                         '"_r": posing a two class classification problem as a regression [-1, 1] problem, '
                         '"_t": transformation is applied (elastic for MNIST and random crop/flip for CIFAR10), '
                         '"_f": (only for cifar10) cifar10_t go through features_nets/cifar10_342f.pt.')
parser.add_argument('--preprocessing', type=str, default='auto', metavar='P',
                    help='P = normalize | squash ... auto,None means normalize is applied')
parser.add_argument('--validation-ratio', type=float, default=0.2, metavar='R',
                    help='Use R * n_train samples from train_data for validation. '
                         'if R = 0 (default): use test_data for validation (to see improvement in test performance)')
parser.add_argument('--expansion', type=int, default=0, metavar='E',
                    help='This option only works for MNIST and CIFAR10 '
                         'If E >= 2, expand the dataset using the appropriate transformation (new_size=E*old_size). '
                         'Note: if transformation is enabled (by using dataset name + _t), and E = 1, the '
                         'transformation is on-the-fly, so each request to the same data item will be returned with'
                         'a different (transformed) item. If E >= 2, the transformation is offline, the dataset is'
                         'E times larger and static, with the first n items identical to original dataset, and the '
                         'next items are their transformation in the same ordering. However, if E>MAX_CACHEABLE (10) '
                         'we simulate online transformation using Ex offline transformations: dataset_size = n, but '
                         'query to image i is returned with one of its E transformed instances.')
parser.add_argument('--n-workers-dl', type=int, default=1, metavar='NW',
                    help='Number of workers for dataloader. None means 0: run in the same process. '
                         '5 means using min(n_cores, 5). -5 means using max(1, n_cores-5). '
                         '-200 means using n_cores//2. -300 means using n_cores//3!')
parser.add_argument('--no-shuffle', action='store_true', default=False,
                    help='Disable shuffling of training set. Normally not required.')
parser.add_argument('--features-net', type=str, default=None, metavar='F',
                    help='Filename containing the features-net: F = None | mnist.pt | cfr10_342f.pt')

# Model params
parser.add_argument('--algorithm', type=str, default='nn', metavar='A',
                    help='Algorithm to be used. A = nn | pfw | lrn | glrn')  # lrn: Lipschitz constrained ResNet
parser.add_argument('--res_sizes', type=str, default='10x100:1.0, 10x100:0.5, 10x100:0.2, 10x100:0.1, 1x100:10',
                    metavar='RS',
                    help='(Only for A=*lrn) Specify the ResNet structure by specifying the size of each '
                         'residual module from input to output. RS = (NxW:O)+. O is dim_out, W is width '
                         'of the residual module (O control how dim_in is gradually reduced to dim_out). '
                         'If O or W is a float, e.g. 0.5, it is relative, e.g. 1/2 of dim_in of the ResNet. '
                         'N is number of consecutive residual modules with sizes O, W.')
parser.add_argument('--model', type=str, default='NN2', metavar='M',
                    help='Specify one of the models of algorithm A. M = NN2 | NN3t | SN4h | Neuron etc.')
parser.add_argument('--activation', type=str, default='relu', metavar='A',
                    help='Activation function. A = relu (default) | sigmoid | tanh | softsign')
parser.add_argument('--bound', type=float, default=1660, metavar='B',
                    help='Specify the bound (scaling) for greedy algorithms, '
                         '0 means automatically set to (4/3)*max, or 10 for classification.'
                         'B is Lipschitz bound of residual module for A=lrn or glrn, '
                         'in which case 0 means unbounded (normal ResNet)')
parser.add_argument('--module-size', type=int, default=100, metavar='S',
                    help='Specify the size of each module. In NN2 it is the number of hidden unit. Not used for A=lrn.')
parser.add_argument('--n-modules', type=int, default=100, metavar='M',
                    help='Specify the total number of modules for greedy algorithms. Not used for A=lrn.')
parser.add_argument('--early-stopping', type=int, default=0, metavar='ES',
                    help='(Only for greedy algorithms) Stop after adding ES modules '
                         'without improving validation performance (0 = Disabled)')
parser.add_argument('--min-n-modules', type=int, default=0, metavar='NE',
                    help='(Only if early-stopping > 0) this specifies the minimum number of modules, '
                         'after which early_stopping starts to be effective.')
parser.add_argument('--filename', type=str, default='', metavar='F',
                    help='Filename of the previously trained model (checkpoint). If specified, the optimizer will '
                         'continue to add more modules to this existing model.')

# Optimization params
parser.add_argument('--loss', type=str, default='nll', metavar='L',
                    help='Loss function. L in (entropy, mae, mse). '
                         'auto means entropy for classification, mse for regression')  # mae = l1  mse = l2
parser.add_argument('--validation-loss', type=str, default='nll', metavar='VL',
                    help='For regression task, use this option to set different loss for training and validation. '
                         'E.g. train loss is mse, but l1 could be used for validation and showing test result. '
                         'For classification this is always error rate. auto means same as L')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='Batch size for training. 0 means auto: 1/4 train size, max 20000.')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='BT',
                    help='Batch size for validation and testing. 0 means auto: 1/4 test size, max 200.')
parser.add_argument('--optimizer', type=str, default='adam', metavar='O',
                    help='Optimization algorithm. O = sgd | nesterov | adam')
parser.add_argument('--n-epochs', type=int, default=0, metavar='E',
                    help='Number of epochs to train. If E=0, optimizer will run until lr is reduced to lr_min')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='Initial learning rate (default: LR = 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: M = 0.9. To disable, set M = 0)')
parser.add_argument('--weight-decay', type=float, default=0.01, metavar='D',
                    help='Weight decay (regularisation) coefficient (default: D = 0)')
parser.add_argument('--min-weight-decay', type=float, default=0, metavar='MD',
                    help='If MD > 0: decrease weight decay coefficient for each module '
                         '(weight decay of the last module = MD, default: MD = 0.0)')
parser.add_argument('--no-bias-decay', action='store_true', default=False,
                    help='This option disable weight decay for biases')
parser.add_argument('--factor', type=float, default=0.1, metavar='F',
                    help='Specify the factor, by which LR is reduced when it needs to be reduced.')
parser.add_argument('--patience', type=int, default=10, metavar='P',
                    help='If loss is not reduced for P consecutive epochs during training, reduce LR <- F*LR.'
                         'If patience=0 --> use exponentialLR (LR <- LR* F^epochs).'
                         'If patience=0 and factor=0 --> use factor F=(min_lr/lr)**(1/epochs), such that lr will'
                         'gradually decreases to min_lr upon the last epoch')
parser.add_argument('--threshold', type=float, default=1e-1, metavar='T',
                    help='When patience > 0, this defines the min gap that loss must be reduced '
                         'after "patience" number of epochs.')
parser.add_argument('--threshold-mode', type=str, default='adp', metavar='M',
                    help='abs (default), rel, or adp. In abs mode, it is required that ' 
                         'new_min_loss < current_min_loss - threshold after patience number of epochs. '
                         'In rel mode, new_min_loss < current_min_loss - threshold * |current_min_loss|.')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='P',
                    help='Minimum learning rate, after which lr is no longer reduced')
parser.add_argument('--milestones', nargs='+', type=int, default=[[0, 1, 80, 120], [0.01, 0.1, 0.01, 0.001]],
                    metavar='N1 N2 ...',
                    help='This has highest priority, if specified, will be used instead of other methods. '
                         '[80, 120] means lr is reduced by factor F at 80 and then 120 epochs. '
                         '[[0, 1, 80, 120], [0.01, 0.1, 0.01, 0.001]] means lr 0.01 for epoch<1 lr 0.1 for epoch<80...')

# warm start parameters, if some parameter is not specified, the value of corresponding general parameter is used
parser.add_argument('--warm-start', type=int, default=0,
                    help='Enable warm start (optimize the first module by SGD (default: False)')
parser.add_argument('--ws-loss', type=str, default='', metavar='WL',
                    help='Loss function for the warm start module. Empty = same as --loss')
parser.add_argument('--ws-validation-loss', type=str, default='', metavar='WVL',
                    help='Validation loss for the warm start module. Empty = same as --validation-loss')
parser.add_argument('--ws-optimizer', type=str, default='', metavar='O',
                    help='Optimization algorithm for warm start module. Empty = same as --optimizer')
parser.add_argument('--ws-batch-size', type=int, default=256, metavar='WB',
                    help='Batch size for training of warm start module (default: B = 256)')
parser.add_argument('--ws-lr', type=float, default=0.01, metavar='WLR',
                    help='Specify the learning rate in warm start phase, '
                         'which can be different from greedy module lr.')
parser.add_argument('--ws-weight-decay', type=float, default=0.0, metavar='WD',
                    help='Specify the weight decay in warm start phase, '
                         'which can be different from greedy module weight decay.')
parser.add_argument('--ws-patience', type=int, default=40, metavar='P',
                    help='If loss is not reduced for P consecutive epochs during training, reduce LR <- F*LR.'
                         'If patience=0 --> use exponentialLR (LR <- LR* F^epochs).'
                         'If patience=0 and factor=0 --> use factor F=(min_lr/lr)**(1/epochs), such that lr will'
                         'gradually decreases to min_lr upon the last epoch')
parser.add_argument('--ws-threshold', type=float, default=1e-3, metavar='T',
                    help='When patience > 0, this defines the min gap that loss must be reduced '
                         'after "patience" number of epochs.')
parser.add_argument('--ws-threshold-mode', type=str, default='rel', metavar='M',
                    help='abs (default), rel, or adp. In abs mode, it is required that ' 
                         'new_min_loss < current_min_loss - threshold after patience number of epochs. '
                         'In rel mode, new_min_loss < current_min_loss - threshold * |current_min_loss|.')
parser.add_argument('--ws-min-lr', type=float, default=1e-6, metavar='P',
                    help='Minimum learning rate, after which lr is no longer reduced')
parser.add_argument('--ws-milestones', nargs='+', type=int, default=None, metavar='N1 N2 ...',
                    help='A list of epochs, at which if specified, LR is reduced by LR <- F*LR')

# Other params
parser.add_argument('--caching', type=int, default=2, metavar='C',
                    help='0: no caching (slow, but mem efficient). 1: (default) cache data (xs, ys), calculate '
                         'other values (zs, zs_grad, losses) on the fly and shuffle items for each epoch. '
                         '2: cache all data (xs, ys, zs, zs_grad, losses) and shuffle all items for each epoch. '
                         '3: same as 2 but only shuffle order of batches. 4: same as 2 but no shuffling (batches and '
                         'their order are static across epochs). Options 3-4 are experimental. Options 1-2 should be '
                         'equivalent (check!) but 2 is much faster. Stick to option 2!')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed, set to make the result reproducible. 0=disabled.')
parser.add_argument('--save-result', type=int, default=1, metavar='SR',
                    help='SR = 0 | 1 | 2 | 3. 0: no saving. 1: save log file. 2: save log and best model. '
                         '3: save log and model checkpoints and best model.')
parser.add_argument('--verbose', type=int, default=6, metavar='V',
                    help='Level of verbose: 0 (silent), 1 (warning), 2 (info), 3 (model:summary), 4 (model:summary2), '
                         '5 (model:details), 6 (debug:epochs), 7 (debug:batches).')

args = parser.parse_args()
print(args.res_sizes)

result = launch(args)
