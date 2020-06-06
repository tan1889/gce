import pickle
import argparse
import os.path
import copy
import random
import launcher
import time
from datetime import datetime
import torch.multiprocessing as mp
import data_utils
import torch
import warnings


class ModelSelection:

    def __init__(self, args, args_grid={}, n_iter=0, n_fold=0, filename=None, n_workers=None, verbose=1):
        """
        args:       run arguments feed to launcher to run the algorithm with some dataset
        args_grid:  {abc: [v1,v2], def: [v3, v4]} tune hyper-parameter abc with values v1 or v2, def with v3, v4
                    {}, None means multiple runs (i_iter) of the same params with different init (args.seed=0 required!)
                    note: values specified by args_grid will override the same ones in args
        n_iter:     0 means grid search (all combinations of args_grid), 100 means random search 100 combinations
        n_fold:     0 means no cross validation, validation_ratio must be > 0 to calculate score
        filename:   filename to record results, if existing -> only not performed experiments will be ran
        n_workers:  how many concurrent processes
        verbose:    0=disable, 1=summary, 2=details, 3=debug
        Examples:
            n_iter=100 n_fold=5 -> random search 100 combinations from hyper params space defined by args_grid
            n_iter=0 n_fold=5 -> grid search all combinations with 5-fold cross validation
            n_iter=0 n_fold=5 -> grid search all combinations no cross validation
            args_grid={} n_iter=100 n_fold=0 args.seed=0 -> 100 repeat runs same parameters, different random seeds
        """

        assert 'dataset' not in args_grid or len(args_grid['dataset']) == 1, \
            'Multiple datasets in args_grid is not allowed!'
        assert 'validation_ratio' not in args_grid or len(args_grid['validation_ratio']) == 1, \
            'Multiple validation_ratio in args_grid is not allowed!'
        assert 'preprocessing' not in args_grid or len(args_grid['preprocessing']) == 1, \
            'Multiple preprocessing in args_grid is not allowed!'
        assert 'seed' not in args_grid or len(args_grid['seed']) == 1, \
            'Multiple seed in args_grid is not allowed!'
        assert 'expansion' not in args_grid or len(args_grid['expansion']) == 1, \
            'Multiple expansion in args_grid is not allowed!'
        assert args_grid or (args.seed == 0 and n_fold == 0 and n_iter > 0), \
            'no args_grid -> multiple runs of the same params -> seed=0 and n_fold=0 required!'

        if isinstance(args_grid, dict):
            for k, vs in args_grid.items():  # complete args with values from args_grid
                setattr(args, k, vs[0])
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        self.verbose = verbose
        if verbose > 3:
            args.verbose = verbose - 3  # control args.verbose by setting verbose > 3, e.g. verbose=8 -> args.verbose=5
        if args.verbose == 0:  # suppress warnings from data_utils
            warnings.simplefilter("ignore")

        launcher.set_auto_params(args)  # if bound, batch_size = 0  -> set them automatically

        self.train_loader, self.val_loader, self.test_loader, self.x_shape, self.out_dim, self.regression = \
            launcher.load_dataset(args)

        _, args.validation_ratio = data_utils.format_exp_valr(1, args.validation_ratio)

        self.args = args
        self.args_grid = args_grid

        self.type = 'grid_search' if n_iter == 0 else 'random_search'
        if not args_grid or len(args_grid) == 0:
            assert n_iter > 0, 'args_grid=None -> repeated run -> n_iter={} must be > 0'.format(n_iter)
            self.type = 'repeated_runs'

        n_cpu = mp.cpu_count()
        if n_workers is None:
            n_workers = 0
        elif n_workers > n_cpu:
            n_workers = n_cpu
        elif n_workers < 0:
            n_workers = max(0, n_cpu + n_workers)
        self.n_workers = n_workers

        if filename:
            self.filename = filename
        else:
            prfx = '{}_{}_{}' if args.desc else '{}{}_{}'
            prfx = prfx.format(args.desc, args.dataset, self.type)
            self.filename = 'output/{}_val{}{}_{:%Y%m%d_%H%M%S}.pkl'.format(
                prfx, args.validation_ratio,
                '_{}-fold'.format(n_fold) if n_fold > 0 else '',
                datetime.now())

        # results is ordered by best first
        if os.path.isfile(self.filename):
            self.results = pickle.load(open(filename, 'rb'))
            if args.dataset != self.results[0]['args'].dataset:
                raise Exception('dataset {} is different from {} in existing results file {}'.format(
                    args.dataset, self.results[0]['args'].dataset, self.filename))
            if args.validation_ratio != self.results[0]['args'].validation_ratio:
                raise Exception('val_ratio {} is different from {} in existing results file {}'.format(
                    args.validation_ratio, self.results[0]['args'].validation_ratio, self.filename))
            if verbose > 0:
                print('Loaded {} existing results from {}'.format(len(self.results), self.filename))
        else:
            self.results = []

        if verbose > 0:
            print('{}   {}   Model: {}.{}'.format(
                self.type.replace('_', ' ').upper(), 'Regression' if self.regression else 'Classification',
                args.algorithm, args.optimizer if args.algorithm == 'misc' else args.model
            ))
            print('DATASET: {} prep={} val={} x{} n_train={} n_val={} n_test={} in_dim={} out_dim={}.'.format(
                args.dataset, args.preprocessing, args.validation_ratio, getattr(args, 'expansion', 0),
                len(self.train_loader), len(self.val_loader), len(self.test_loader), self.x_shape, self.out_dim
            ))
            print('Generating args ({})'.format(self.type))

        n_done = 0
        if self.results and (self.type == 'repeated_runs' or self.type == 'random_search'):
            if not hasattr(self.results[0], 'input_args'):
                n_done = len(self.results)
            else:
                for r in self.results:
                    if r['input_args'] == args and r['input_args_grid'] == args_grid and r['input_n_fold'] == n_fold:
                        n_done += 1

        self.args_list = []
        if self.type == 'repeated_runs':
            self.args_list = [copy.deepcopy(args) for _ in range(max(0, n_iter - n_done))]
            if verbose > 0:
                print('   Generated {} repeated runs of the same args'.format(len(self.args_list)))
        else:
            if self.type == 'random_search':
                if n_iter - n_done > 0:  # continue to run the remaining number of tests
                    self.args_list = self.generate_args_list(args, args_grid, n_iter - n_done, self.results, verbose)
            else:  # grid_search
                self.args_list = self.generate_args_list(args, args_grid, n_iter, self.results, verbose)
            if verbose > 0:
                print('   Generated {} non-duplicating experiments'.format(len(self.args_list)))

        assert n_fold <= 0 or args.validation_ratio == 0.0, \
            'n_fold={} -> cross validation -> val_ratio={}, must be 0!'.format(n_fold, args.validation_ratio)
        self.n_fold = n_fold
        self.fold_ids = data_utils.get_fixed_randperm(len(self.train_loader.dataset))

    @staticmethod
    def same_args(a, a1):
        ignored = {'desc', 'n_workers_dl', 'n_workers', 'device', 'no_cuda', 'cuda', 'save_result',
                   'verbose', 'test_batch_size', 'regression'}

        def reduce_prep(prep):
            if not prep:
                return ''
            prep = prep.replace('auto', '').replace('default', '').replace('normalize', '')
            prep = prep.replace(', regression', '').replace('transform, ', '')  # differentiate by dataset name
            return prep

        for k in vars(a):
            if k == 'preprocessing':
                if reduce_prep(getattr(a, k)) != reduce_prep(getattr(a1, k)):
                    return False
            elif k == 'dataset':
                if getattr(a, k).lower() != getattr(a1, k).lower():
                    return False
            elif getattr(a, k) != getattr(a1, k) and k not in ignored:
                return False
        return True

    @staticmethod
    def generate_args_list(args, args_grid, n_iter, results, verbose):

        def tested(a):  # if args a is already tested (in the list of args of results)
            for r in results:
                if ModelSelection.same_args(a, r['args']):
                    return True
            return False

        if not args_grid: return [args]

        dataset = args.dataset
        val_ratio = args.validation_ratio
        args = copy.deepcopy(args)
        args_list = []
        if n_iter <= 0:  # grid search
            params = list(args_grid.keys())
            ids = [0 for _ in range(len(params) + 1)]
            n_dupl = 0
            for i in range(len(params)):
                setattr(args, params[i], args_grid[params[i]][0])
            while ids[-1] == 0:
                i = 0
                if ids[i] < len(args_grid[params[i]]):
                    setattr(args, params[i], args_grid[params[i]][ids[i]])
                    if args not in args_list:
                        if tested(args):
                            n_dupl += 1
                        else:
                            assert args.dataset == dataset, 'dataset should be the same!'
                            assert args.validation_ratio == val_ratio, 'validation_ratio should be the same!'
                            args_list.append(copy.deepcopy(args))
                    ids[i] += 1
                else:
                    while ids[-1] == 0 and ids[i] >= len(args_grid[params[i]]):
                        ids[i] = 0
                        ids[i + 1] += 1
                        setattr(args, params[i], args_grid[params[i]][0])
                        i += 1
                    if ids[-1] == 0:
                        setattr(args, params[i], args_grid[params[i]][ids[i]])
            if n_dupl > 0 and verbose > 0:
                print('   Ignored {} args sets (already exist in the report file)!'.format(n_dupl))
        else:
            c = 0  # count number of unsuccessful samples, quit if it is too large (exhausted param space)
            while len(args_list) < n_iter and c < 1000:
                for param, values in args_grid.items():
                    value = random.choice(values)
                    setattr(args, param, value)
                c += 1
                if args not in args_list and not tested(args):
                    assert args.dataset == dataset, 'dataset should be the same!'
                    assert args.validation_ratio == val_ratio, 'validation_ratio should be the same!'
                    args_list.append(copy.deepcopy(args))
                    c = 0
            if c >= 1000 and verbose > 0:
                print('   Quited generating random args: the search space given by args_grid is exhausted!')
        return args_list

    @staticmethod
    def elapsed_time(t):
        if t < 1:  # milli seconds
            return '{:.1f} ms'.format(t*1000)
        elif t < 60:  # seconds
            return '{:.2f} sec'.format(t)
        else:  # minutes
            return '{:.2f} min'.format(t / 60)

    @staticmethod
    def get_index(results, r):  # find position of r in ordered list results
        if len(results) == 0: return 0
        if len(results) == 1: return int(results[0]['validation_score'] <= r['validation_score'])
        i, j = 0, len(results) - 1
        while i < j - 1:
            k = (i + j) // 2
            if results[k]['validation_score'] <= r['validation_score']:
                i = k
            else:
                j = k
        if results[j]['validation_score'] <= r['validation_score']: return j + 1
        elif results[i]['validation_score'] > r['validation_score']: return i
        else: return i + 1

    @staticmethod
    def mean_dict(dlist, key=None):
        """dlist: list of dictionaries of same keys with numeric values, return a dict of mean of the values"""
        r = dict()
        if key is None:
            for k, v in dlist[0].items():
                if isinstance(v, dict):
                    r[k] = ModelSelection.mean_dict(dlist, key=k)
                elif isinstance(v, int):
                    r[k] = sum([dlist[i][k] for i in range(len(dlist))]) // len(dlist)
                elif isinstance(v, float):
                    r[k] = sum([dlist[i][k] for i in range(len(dlist))]) / len(dlist)
        else:
            for k, v in dlist[0][key].items():
                if isinstance(v, int):
                    r[k] = sum([dlist[i][key][k] for i in range(len(dlist))]) // len(dlist)
                elif isinstance(v, float):
                    r[k] = sum([dlist[i][key][k] for i in range(len(dlist))]) / len(dlist)
        return r

    def get_fold_ids(self, fold):
        assert 0 <= fold < self.n_fold, 'Invalid fold={}. Must be 0 <= fold < {}'.format(fold, self.n_fold)
        n_train = self.fold_ids.shape[0]
        size = n_train // self.n_fold
        i0 = fold * size
        i1 = i0 + size
        if fold == self.n_fold - 1:
            i1 = n_train
        val_ids = self.fold_ids[i0:i1]

        if i0 == 0:
            train_ids = self.fold_ids[i1:]
        elif i1 == n_train:
            train_ids = self.fold_ids[:i0]
        else:
            train_ids = torch.cat([self.fold_ids[:i0], self.fold_ids[i1:]])

        return train_ids, val_ids

    def result_str(self, r):
        if self.regression:
            s = 'validation_score={:.4f}   n_modules={}\n      train: rmse={:.4f} mae={:.4f}' \
                '\n      valdt: rmse={:.4f} mae={:.4f}\n      test : rmse={:.4f} mae={:.4f}'.format(
                    r['validation_score'], r['best_n_modules'],
                    r['best_train']['train_rmse'], r['best_train']['train_mae'],
                    r['best_validation']['val_rmse'], r['best_validation']['val_mae'],
                    r['test']['rmse'], r['test']['mae'])
        else:
            s = 'validation_score={:.4f}   n_modules={}   train={:.4f} val={:.4f} test={:.4f}'.format(
                    r['validation_score'], r['best_n_modules'],
                    r['best_train']['train_error'],
                    r['best_validation']['val_error'],
                    r['test']['error'])

        if 'n_fold' in r:
            s += '\n      ({}-fold val_scores=[{}])'.format(
                r['n_fold'],
                ', '.join(['{:.4f}'.format(i['validation_score']) for i in r['fold_results']]))
        return s

    def run_experiment(self, args):

        try:
            if self.n_fold <= 0:  # no cross validation
                # reuse dataloader to share underlying dataset -> save memory
                train_loader = self.train_loader.new(batch_size=args.batch_size, shuffle=not args.no_shuffle)
                val_loader = self.val_loader.new(batch_size=args.test_batch_size)
                test_loader = self.test_loader.new(batch_size=args.test_batch_size)

                result = launcher.launch(
                    args, (train_loader, val_loader, test_loader, self.x_shape, self.out_dim, self.regression))

            else:  # k-fold cross validation
                fold_results = []
                test_loader = self.test_loader.new(batch_size=args.test_batch_size)
                args.regression = self.regression
                for k in range(self.n_fold):
                    t0 = time.time()
                    train_ids, val_ids = self.get_fold_ids(k)
                    train_loader = self.train_loader.subset(train_ids, batch_size=args.batch_size,
                                                            shuffle=not args.no_shuffle)
                    val_loader = self.train_loader.subset(val_ids, batch_size=args.test_batch_size)

                    fold_result = launcher.launch(
                        args, (train_loader, val_loader, test_loader, self.x_shape, self.out_dim, self.regression))

                    fold_results.append(fold_result)

                result = self.mean_dict(fold_results)
                result['args'] = args
                result['n_fold'] = self.n_fold
                result['fold_results'] = fold_results

        except (AssertionError, RuntimeError, IOError, OverflowError, MemoryError) as error:
            print('Error:', error, '\nAborting args=', args)
            result = {'validation_score': float('inf')}

        result['input_args'] = self.args
        result['input_args_grid'] = self.args_grid
        result['input_n_fold'] = self.n_fold
        return result

    def train(self):
        n_jobs = len(self.args_list)
        n0 = len(self.results)

        if self.verbose > 0:
            if self.n_workers > 0:
                print('Scheduling {} jobs with {} processes'.format(n_jobs, self.n_workers))
            else:
                print('Executing {} jobs sequentially'.format(n_jobs))

        t0 = time.time()
        print_interval = max(1, len(self.args_list) // 5)

        if self.n_workers == 0:  # exec in main process
            for args in self.args_list:
                r = self.run_experiment(args)
                if r['validation_score'] < float('inf'):
                    i = self.get_index(self.results, r)

                    self.results.insert(i, r)  # self.results is always sorted best first
                    pickle.dump(self.results, open(self.filename, "wb"))

                    if self.verbose > 1 or (self.verbose > 0 and len(self.results) % print_interval == 0):
                        print('   {}/{} jobs finished in {}.\n   Best: {}'.format(
                            len(self.results) - n0, n_jobs, self.elapsed_time(time.time() - t0),
                            self.result_str(self.results[0])))
                    if i == 0:  # found new best
                        pickle.dump(self.results, open(self.filename, "wb"))
                        if self.verbose > 2:
                            print('   New best:', self.result_str(r))
                            print('\t\targs={}'.format(r['args']))
                    elif self.verbose > 2:
                        print('   New result:', self.result_str(r))
                        print('\t\targs={}'.format(r['args']))
        else:  # use worker processes
            with mp.Pool(processes=self.n_workers) as pool:
                # with MyPool(processes=self.n_workers) as pool:
                for r in pool.imap_unordered(self.run_experiment, self.args_list):
                    if r['validation_score'] < float('inf'):
                        i = self.get_index(self.results, r)

                        self.results.insert(i, r)  # self.results is always sorted best first
                        pickle.dump(self.results, open(self.filename, "wb"))

                        if self.verbose > 1 or (self.verbose > 0 and len(self.results) % print_interval == 0):
                            print('   {}/{} jobs finished in {}.\n   Best: {}'.format(
                                      len(self.results) - n0, n_jobs, self.elapsed_time(time.time() - t0),
                                      self.result_str(self.results[0])))
                        if i == 0:  # found new best
                            pickle.dump(self.results, open(self.filename, "wb"))
                            if self.verbose > 2:
                                print('   New best:', self.result_str(r))
                                print('\t\targs={}'.format(r['args']))

        if self.verbose > 0:
            print('All {} jobs finished in {}'.format(n_jobs, self.elapsed_time(time.time() - t0)))
            print('Best', self.result_str(self.results[0]))
            if self.verbose > 1:
                print('args:', self.results[0]['args'])
            print('Result saved to:', self.filename)
        return self.results[0]


# ==================== TEST =================================================================== #

if __name__ == '__main__':
    ar = argparse.Namespace(desc='test', dataset='digits', preprocessing='normalize',
                            validation_ratio=0.2, expansion=0, no_shuffle=False,
                            batch_size=128, test_batch_size=128,
                            algorithm='misc', optimizer='rforest',
                            criterion='gini', n_estimators=100,
                            early_stopping=50, min_n_estimators=10,
                            max_features='auto', max_depth=10,
                            min_samples_split=5, min_samples_leaf=5, no_bootstrap=False,
                            save_result=0, no_cuda=True, num_workers=1, seed=1, verbose=1)
    ar_grid = {'n_estimators': [100, 200, 300, 500, 700, 1000],
               'max_features': ['auto', 'sqrt', 'log2', 2],
               'max_depth': [x for x in range(10, 101, 10)],
               'min_samples_split': [3, 5, 7, 9, 11],
               'min_samples_leaf': [3, 5, 7, 9, 11],
               'no_bootstrap': [True, False]}

    ms = ModelSelection(ar, ar_grid, verbose=2, n_iter=10, n_workers=2, filename='../output/test_modelselection2.pkl')
    res = ms.train()
