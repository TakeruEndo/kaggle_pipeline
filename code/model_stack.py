# import os
import pandas as pd
import numpy as np

# from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
# from sklearn.svm import SVRz
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.svm import SVC

SEED = 0

train = pd.read_csv("../data/raw/train_2.csv")
train_x = train.drop(['SalePrice'], axis=1)
train_y = train['SalePrice']
test = pd.read_csv("../data/raw/test_2.csv")


# Sclearn分類機を拡張
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        if clf == XGBRegressor or clf == LGBMRegressor:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            self.clf = clf(**params, early_stopping_rounds=early_stopping_rounds)
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


class ModelStack():

    def __init__(self, setting, params):
        train = pd.read_csv("../data/raw/train_2.csv")
        self.train_x = train.drop(['SalePrice'], axis=1)
        self.train_y = train['SalePrice']
        self.test = pd.read_csv("../data/raw/test_2.csv")
        self.taregt = setting['target']
        self.params = params

    def predict_cv(self, model, train_x, train_y, test_x):
        preds = []
        preds_test = []
        va_idxes = []
        kf = KFold(n_splits=3, shuffle=True, random_state=71)
        # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
        for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
            tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
            tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
            model.fit(tr_x, tr_y)
            pred = model.predict(va_x)
            preds.append(pred)
            pred_test = model.predict(test_x)
            preds_test.append(pred_test)
            va_idxes.append(va_idx)
        # バリデーションに対する予測値を連結して、そのあと元の順序に並べなおす
        va_idxes = np.concatenate(va_idxes)
        preds = np.concatenate(preds, axis=0)
        order = np.argsort(va_idxes)
        pred_train = preds[order]
        # テストデータに対する予測値の平均をとる
        preds_test = np.mean(preds_test, axis=0)
        return pred_train, pred_test

    def train(self):
        
        # モデルの用意
        # 5つの学習モデルのオブジェクトを作成
        rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=self.params['rf_params'])
        et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=self.params['et_params'])
        ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=self.params['ada_params'])
        gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=self.params['gb_params'])
        svc = SklearnHelper(clf=SVC, seed=SEED, params=self.params['svc_params'])
        lightgbm = SklearnHelper(clf=LGBMRegressor, seed=SEED, params=self.params['lgb_params'])
        xgboost = SklearnHelper(clf=XGBRegressor, seed=SEED, params=self.params['xgb_params'])

        # 1層目のモデル
        # pred_train_1a, pred_train_1bは、学習データのクロスバリデーションでの予測値
        # pred_test_1a, pred_test_1bは、テストデータの予測値
        model_1a = xgboost
        pred_train_1a, pred_test_1a = self.predict_cv(model_1a, train_x, train_y, test)

        model_1b = rf
        pred_train_1b, pred_test_1b = self.predict_cv(model_1b, train_x, train_y, test)

        # 1層目のモデルの評価
        print(f'mean_squared_error: {mean_squared_error(train_y, pred_train_1a):.4f}')
        print(f'mean_squared_error: {mean_squared_error(train_y, pred_train_1b):.4f}')

        # 予測値を特徴量としてデータフレームを作成
        train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
        test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

        # 2層目のモデル
        # pred_train_2は、２層目のモデルの学習データのクロスバリデーションの予測値
        # pred_test_2は、２層目のモデルのテストデータの予測値
        model_2 = lightgbm
        pred_train_2, pred_test_2 = self.predict_cv(model_2, train_x_2, train_y, test_x_2)
        print(f'mean_squared_error: {mean_squared_error(train_y, pred_train_2):.4f}')

        return pred_test_2

