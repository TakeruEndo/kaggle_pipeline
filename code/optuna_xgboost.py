"""
Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.
In this example, we optimize the validation accuracy of cancer detection
using XGBoost. We optimize both the choice of booster model and their hyper
parameters.
We have following two ways to execute this example:
(1) Execute this code directly.
    $ python xgboost_simple.py
(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize xgboost_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db
"""

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

import optuna


class Objective:
    """目的関数に相当するクラス"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        train_x, test_x, train_y, test_y = train_test_split(self.X, self.y, test_size=0.25)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)
        
        param = {
            'objective': 'reg:squarederror',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
        }

        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dtest)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
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