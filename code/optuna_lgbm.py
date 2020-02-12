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
import yaml
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)

MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_PATH']


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


def load_x_train(features) -> pd.DataFrame:
    """学習データの特徴量を読み込む
    列名で抽出する以上のことを行う場合、このメソッドの修正が必要
    :return: 学習データの特徴量
    """
    # 学習データの読込を行う
    dfs = [pd.read_pickle(FEATURE_DIR_NAME + f'{f}_train.pkl') for f in features]
    df = pd.concat(dfs, axis=1)

    # 特定の値を除外して学習させる場合 -------------
    # self.remove_train_index = df[(df['age']==64) | (df['age']==66) | (df['age']==67)].index
    # df = df.drop(index = self.remove_train_index)
    # -----------------------------------------
    return df


def load_y_train(target) -> pd.Series:
    """学習データの目的変数を読み込む
    対数変換や使用するデータを削除する場合には、このメソッドの修正が必要
    :return: 学習データの目的変数
    """

    # 目的変数の読込を行う
    train_y = pd.read_pickle(FEATURE_DIR_NAME + target + '_train.pkl')

    # 特定の値を除外して学習させる場合 -------------
    # train_y = train_y.drop(index = self.remove_train_index)
    # -----------------------------------------

    return pd.Series(train_y[target])


if __name__ == '__main__':

    # pklからロードする特徴量の指定
    features = [
        'acceleration',
        'car_label_encoder',
        'cylinders',
        'displacement',
        # 'displacement_plus_horsepower',
        'horsepower',
        'model_year',
        'origin',
        'power',
        'weight',
    ]

    target = 'mpg'

    X = load_x_train(features)
    y = load_y_train(target)

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
