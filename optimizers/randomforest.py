import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import multiprocessing


class RandomForest:

    def __init__(self, train_loader, val_loader, x_shape, dim_out, args):
        assert 0 < args.min_n_estimators <= args.n_estimators and args.step_size > 0, \
            'Invalid n_estimators setting. Required 0<min<=max and step > 0.'

        self.args = args
        self.x_train, self.y_train = train_loader.numpy_data()
        self.x_val, self.y_val = val_loader.numpy_data()

        if len(self.x_train.shape) > 2:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
            self.x_val = self.x_val.reshape(self.x_val.shape[0], -1)

        assert args.criterion in 'auto, entropy, gini, mae, mse', \
            'args.criterion={} not in (entropy, gini, mae, mse)'.format(args.criterion)
        criterion = args.criterion
        if criterion == 'auto':
            criterion = 'mse' if args.regression else 'gini'

        assert (args.regression and args.criterion in {'auto', 'mse', 'mae'}) \
            or (not args.regression and args.criterion in {'auto', 'gini', 'entropy'}), \
            '{} loss is not for {}'.format(args.criterion, 'regression' if args.regression else 'classification')

        max_features = args.max_features
        if isinstance(max_features, str):
            if str.isdigit(max_features):
                max_features = int(max_features)
            elif max_features not in {'auto', 'sqrt', 'log2'}:
                max_features = float(max_features)
        if isinstance(max_features, float):
            if max_features < 0.: max_features = 0.
            if max_features > 1.: max_features = 1.
        if isinstance(max_features, int):
            if max_features < 0: max_features = 0
            if max_features > self.x_train.shape[1]: max_features = self.x_train.shape[1]

        num_cpu = multiprocessing.cpu_count()
        if not hasattr(args, 'n_workers') or args.n_workers is None:
            args.n_workers = 1
        elif args.n_workers > num_cpu:
            args.n_workers = num_cpu
            if args.verbose >= 3:
                print('args.n_workers is inefficiently large, changed it to num_cpu=' + str(num_cpu))
        elif args.n_workers <= -300:
            args.n_workers = max(1, num_cpu // 3)
        elif args.n_workers <= -200:
            args.n_workers = max(1, num_cpu // 2)
        elif args.n_workers < 0:
            args.n_workers = max(1, num_cpu + args.n_workers)

        self.params = {'criterion': criterion,
                       'n_estimators': args.n_estimators,
                       'max_features': max_features,
                       'max_depth': args.max_depth,
                       'min_samples_split': args.min_samples_split,
                       'min_samples_leaf': args.min_samples_leaf,
                       'bootstrap': not args.no_bootstrap,
                       'random_state': args.seed if args.seed > 0 else None,
                       'n_jobs': args.n_workers,
                       'verbose': args.verbose - 3 if args.verbose > 3 else 0
                       }

        self.result = {'train_loss': float('inf'), 'train_error': float('inf'),
                       'val_loss': float('inf'), 'val_error': float('inf'), 'n_estimators': 0}
        self.model = None

        if args.verbose >= 3:
            print('RANDOM FOREST OPTIMIZER LOADED: ')

    def eval(self, estimator, x, y):
        pred = estimator.predict(x)

        mse = rmse = mae = error = float('inf')
        if self.args.regression:
            mse = (np.sum((pred - y)**2) / y.shape[0]).item()
            rmse = mse**0.5
            mae = (np.sum(np.abs(pred - y)) / y.shape[0]).item()
        else:
            correct = np.sum(pred == y).item()
            error = 1.0 - correct / y.shape[0]
        loss = mse if self.params['criterion'] == 'mse' else mae
        return loss, error, rmse, mae

    def train(self):

        if self.args.early_stopping <= 0:  # no early stopping
            rf = RandomForestRegressor(**self.params) if self.args.regression else RandomForestClassifier(**self.params)
            rf.fit(self.x_train, self.y_train)
        else:  # determine best number of trees by adding in trees iteratively
            self.params['n_estimators'] = self.args.min_n_estimators
            self.params['warm_start'] = True
            rf = RandomForestRegressor(**self.params) if self.args.regression else RandomForestClassifier(**self.params)

            result = {'val_loss': float('inf'), 'val_error': float('inf'), 'n_estimators': 0}

            while rf.n_estimators < self.args.n_estimators:
                rf.fit(self.x_train, self.y_train)
                vl_loss, vl_error, _, _ = self.eval(rf, self.x_val, self.y_val)

                if self.args.verbose >= 3:
                    print('   n_estimators={}   validation_score={:.4f}'.format(
                        rf.n_estimators, vl_loss if self.args.regression else vl_error))

                if (self.args.regression and vl_loss < result['val_loss']) \
                        or (not self.args.regression and vl_error < result['val_error']):
                    result = {'val_loss': vl_loss, 'val_error': vl_error, 'n_estimators': rf.n_estimators}
                elif rf.n_estimators - self.args.early_stopping >= result['n_estimators']:
                    rf.set_params(n_estimators=result['n_estimators'])  # roll back rf to optimal n_estimators
                    rf.estimators_ = rf.estimators_[:rf.n_estimators]
                    self.params['n_estimators'] = result['n_estimators']
                    if self.args.verbose >= 3:
                        print('Early stopping at n_estimators={} (added {} trees with no improvement).'.format(
                            rf.n_estimators, self.args.early_stopping))
                        print('Rolled back to optimal n_estimators={}'.format(rf.n_estimators))
                    break

                rf.set_params(n_estimators=rf.n_estimators + self.args.step_size)

        tr_loss, tr_error, tr_rmse, tr_mae = self.eval(rf, self.x_train, self.y_train)
        vl_loss, vl_error, vl_rmse, vl_mae = self.eval(rf, self.x_val, self.y_val)
        self.model = rf
        self.params['n_estimators'] = rf.n_estimators
        self.result = {'train_loss': tr_loss, 'train_error': tr_error, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
                       'val_loss': vl_loss, 'val_error': vl_error, 'val_rmse': vl_rmse, 'val_mae': vl_mae,
                       'n_estimators': rf.n_estimators}

        if self.args.verbose >= 3:
            if self.args.regression and not self.args.dataset.endswith("_r"):
                print('TRAIN RESULT:   Loss: {:.5f}   RMSE: {:.5f}   MAE: {:.5f}'.format(tr_loss, tr_rmse, tr_mae))
                print('VALDT RESULT:   Loss: {:.5f}   RMSE: {:.5f}   MAE: {:.5f}'.format(vl_loss, vl_rmse, vl_mae))
            else:
                print('TRAIN RESULT:   Loss: {:.5f}   Error: {:.2f}%   Accuracy: {:.2f}%'.format(
                    tr_loss, 100. * tr_error, 100. * (1 - tr_error)))
                print('VALDT RESULT:   Loss: {:.5f}   Error: {:.2f}%   Accuracy: {:.2f}%'.format(
                    vl_loss, 100. * vl_error, 100. * (1 - vl_error)))

    def test(self, dataloader):
        assert self.model, 'Model is not yet trained. Train it first!'
        x_test, y_test = dataloader.numpy_data()
        if len(x_test.shape) > 2:
            x_test = x_test.reshape(x_test.shape[0], -1)
        return self.eval(self.model, x_test, y_test)

    def predict(self, x):
        assert self.model, 'Model is not yet trained. Train it first!'
        return self.model.predict(x)

    @property
    def best_n_modules(self):  # number of modules/estimators in the final best model
        return self.result['n_estimators']
