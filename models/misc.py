from optimizers.randomforest import RandomForest
from optimizers.xgboost import XGBoost


class Model:
    def __init__(self, train_loader, val_loader, x_shape, dim_out, args):
        assert args.optimizer in {'xgb', 'xgboost', 'rf', 'rforest', 'randomforest'}
        self.args = args

        if args.optimizer in {'xgb', 'xgboost'}:
            self.optimizer = XGBoost(train_loader, val_loader, x_shape, dim_out, args)
        elif args.optimizer in {'rf', 'rforest', 'randomforest'}:
            self.optimizer = RandomForest(train_loader, val_loader, x_shape, dim_out, args)

        if args.verbose >= 3:
            print('\n{}\tds={}  prep={}  val_ratio={}  alg={}  criterion={}  n_estimators={}  early_stopping={}'.format(
                'REGRESSION:' if args.regression else 'CLASSIFICATION:',
                args.dataset, args.preprocessing, args.validation_ratio, args.optimizer, args.criterion,
                args.n_estimators, args.early_stopping))

    def train(self):
        self.optimizer.train()
        best_train = self.optimizer.result.copy()
        best_val = self.optimizer.result.copy()
        best_train['module'] = best_train['n_estimators']
        best_val['module'] = best_val['n_estimators']
        best_train['epoch'] = best_val['epoch'] = 0
        log = best_val.copy()
        log['log'] = []
        return best_train, best_val, [log]

    def test(self, test_loader):
        return self.optimizer.test(test_loader)

    @property
    def best_validation_model(self):
        return self.optimizer.model

    @property
    def best_train_model(self):
        return self.optimizer.model

    @property
    def best_n_modules(self):  # number of modules/estimators in the final best model
        return self.optimizer.best_n_modules
