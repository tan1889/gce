import numpy as np
import multiprocessing
import xgboost as xgb  # requires xgboost package, installed e.g. via 'pip install xgboost'


class XGBoost:

    def __init__(self, train_loader, val_loader, x_shape, dim_out, args):
        self.args = args
        self.dim_out = dim_out

        if args.regression:
            objective = 'reg:linear'
            eval_metric = 'rmse'
            if args.criterion in {'mae', 'l1'}:
                eval_metric = 'mae'
            elif args.criterion not in {None, 'auto', 'rmse'}:
                raise Exception('Unknown eval_metric={}. For regression, use auto (rmse) | mae (l1).'.format(
                    args.criterion))
        else:
            if self.dim_out > 2:
                objective = 'multi:softmax'  # out 1 vector of classes
                if args.criterion in {None, 'auto', 'error', 'merror'}:
                    eval_metric = 'merror'
                elif args.criterion in {'logloss', 'nll'}:
                    eval_metric = 'mlogloss'
                else:
                    raise Exception('eval_metric={} is not supported for multi-classes classification. '
                                    'Use auto (merror) | logloss (nll)'.format(args.criterion))
            else:
                objective = 'binary:hinge'  # 'binary:logistic'  # logistic -> predict outputs probability, not class
                if args.criterion in {None, 'auto', 'error', 'merror'}:
                    eval_metric = 'error'
                elif args.criterion in {'logloss', 'nll'}:  # auc somehow only works with 2 classes
                    eval_metric = 'logloss'
                elif args.criterion == 'auc':  # auc somehow only works with 2 classes
                    eval_metric = 'auc'
                else:
                    raise Exception('eval_metric={} is not supported for 2-class classification. '
                                    'Use auto (error) | logloss (nll) | auc'.format(args.criterion))

        self.x_train, self.y_train = train_loader.numpy_data()
        self.x_val, self.y_val = val_loader.numpy_data()
        if len(self.x_train.shape) > 2:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
            self.x_val = self.x_val.reshape(self.x_val.shape[0], -1)
        self.dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        self.dval = xgb.DMatrix(self.x_val, label=self.y_val)

        if args.early_stopping is not None and args.early_stopping <= 0:
            self.early_stopping = None
        else:
            self.early_stopping = args.early_stopping

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

        self.params = {'objective': objective,
                       'eval_metric': eval_metric,
                       'seed': args.seed,
                       'max_depth': args.max_depth,
                       'eta': args.eta,
                       'min_child_weight': args.min_child_weight,
                       'gamma': args.gamma,
                       'subsample': args.subsample,
                       'colsample_bytree': args.colsample_bytree,
                       'lambda': args.reg_lambda,
                       'alpha': args.reg_alpha,
                       'scale_pos_weight': args.scale_pos_weight,
                       'nthread': args.n_workers}
        if objective == 'multi:softmax' or objective == 'multi:softprob':
            self.params['num_class'] = dim_out

        self.result = {'n_estimators': 0}
        self.model = None

        if args.verbose >= 3:
            print('XGBOOST OPTIMIZER LOADED: ')

    def eval(self, x, y):
        pred = self.predict(x)

        rmse = mae = error = float('inf')
        if self.args.regression:
            rmse = np.sqrt(np.sum((pred - y)**2) / y.shape[0]).item()
            mae = (np.sum(np.abs(pred - y)) / y.shape[0]).item()
        else:
            if self.params['objective'] == 'binary:logistic':  # pred is probability, not  class
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                pred = pred.astype(int)
            correct = np.sum(pred == y).item()
            error = 1.0 - float(correct) / y.shape[0]
        loss = rmse if self.params['eval_metric'] == 'rmse' else mae
        return loss, error, rmse, mae

    def train(self):
        eval_list = [(self.dtrain, 'train'), (self.dval, 'valdt')]  # last in the list is used for early stopping
        if self.args.verbose >= 5:
            verbose_eval = True
        elif self.args.verbose <= 3:
            verbose_eval = False
        else:
            verbose_eval = self.args.n_estimators // 10  # output only 10 evaluation info

        self.model = xgb.train(self.params, self.dtrain, self.args.n_estimators, eval_list,
                               verbose_eval=verbose_eval, early_stopping_rounds=self.early_stopping)

        tr_loss, tr_error, tr_rmse, tr_mae = self.eval(self.x_train, self.y_train)
        vl_loss, vl_error, vl_rmse, vl_mae = self.eval(self.x_val, self.y_val)

        self.result = {'train_loss': tr_loss, 'train_error': tr_error, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
                       'val_loss': vl_loss, 'val_error': vl_error, 'val_rmse': vl_rmse, 'val_mae': vl_mae,
                       'n_estimators': self.model.best_ntree_limit}

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
        x_test, y_test = dataloader.numpy_data()
        if len(x_test.shape) > 2:
            x_test = x_test.reshape(x_test.shape[0], -1)
        return self.eval(x_test, y_test)

    def predict(self, x):
        assert self.model, 'model is not yet trained. Call train() first!'
        x = xgb.DMatrix(x)
        pred = self.model.predict(x, ntree_limit=self.model.best_ntree_limit)
        return pred

    @property
    def best_n_modules(self):
        return self.result['n_estimators']
