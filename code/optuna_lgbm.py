"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.
In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.
We have following two ways to execute this example:
(1) Execute this code directly.
    $ python lightgbm_simple.py
(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize lightgbm_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db
"""

import lightgbm as lgb
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


class Objective:
    """目的関数に相当するクラス"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        """オブジェクトが呼び出されたときに呼ばれる特殊メソッド"""
        train_x, test_x, train_y, test_y = train_test_split(self.X, self.y, test_size=0.25)
        dtrain = lgb.Dataset(train_x, label=train_y)

        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth', -3, -1),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.01),
            # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        gbm = lgb.train(param, dtrain)
        preds = gbm.predict(test_x)
        accuracy = sklearn.metrics.mean_squared_error(test_y, preds)
        return accuracy


if __name__ == '__main__':

    train = pd.read_csv("../data/raw/train_2.csv")
    X = train.drop(['SalePrice'], axis=1)
    y = train['SalePrice']

    objective = Objective(X, y)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
