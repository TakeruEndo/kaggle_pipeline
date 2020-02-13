import sys
import os
import shutil
import datetime
import yaml
import json
import collections as cl
import warnings
import pandas as pd
import numpy as np
from model_lgb import ModelLGB
from model_xgb import ModelXGB
from model_cb import ModelCB
from model_stack import ModelStack
from runner import Runner
from util import Submission

warnings.filterwarnings("ignore")
now = datetime.datetime.now()
suffix = now.strftime("_%m%d_%H%M")
warnings.simplefilter('ignore')
key_list = ['load_features', 'use_features', 'model_params', 'cv', 'dataset']

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.load(file)
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME']
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_PATH']


def exist_check(path, run_name):
    """学習ファイルの存在チェックと実行確認
    """
    dir_list = []
    for d in os.listdir(path):
        dir_list.append(d.split('-')[-1])

    if run_name in dir_list:
        print('同名のrunが実行済みです。再実行しますか？[Y/n]')
        x = input('>> ')
        if x != 'Y':
            print('終了します')
            sys.exit(0)

    # 通常の実行確認
    print('特徴量ディレクトリ:{} で実行しますか？[Y/n]'.format(FEATURE_DIR_NAME))
    x = input('>> ')
    if x != 'Y':
        print('終了します')
        sys.exit(0)


def my_makedirs(path):
    """引数のpathディレクトリが存在しなければ、新規で作成する
    path:ディレクトリ名
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def my_makedirs_remove(path):
    """引数のpathディレクトリを新規作成する（存在している場合は削除→新規作成）
    path:ディレクトリ名
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def save_model_config(key_list, value_list, dir_name, run_name):
    """jsonファイル生成
    """
    ys = cl.OrderedDict()
    for i, v in enumerate(key_list):
        data = cl.OrderedDict()
        data = value_list[i]
        ys[v] = data

    fw = open(dir_name + run_name + '_param.json', 'w')
    json.dump(ys, fw, indent=4, default=set_default)


def set_default(obj):
    """json出力の際にset型のオブジェクトをリストに変更する
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


if __name__ == '__main__':

    # pklからロードする特徴量の指定
    features = []

    # CVの設定.methodは[KFold, StratifiedKFold ,GroupKFold]から選択可能
    # CVしない場合（全データで学習させる場合）はmethodに'None'を設定
    # StratifiedKFold or GroupKFoldの場合は、cv_targetに対象カラム名を設定する
    cv = {
        'method': 'KFold',
        'n_splits': 3,
        'random_state': 42,
        'shuffle': True,
        # 'cv_target': 'partner_area_cluster'
    }

    # ######################################################
    # 学習・推論 LightGBM ###################################

    # run nameの設定
    run_name = 'lgb'
    run_name = run_name + suffix
    dir_name = MODEL_DIR_NAME + run_name + '/'

    exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # 諸々の設定
    setting = {
        'run_name': run_name,  # run名
        'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'target': 'SalePrice',  # 目的変数
        'calc_shap': False,  # shap値を計算するか否か
        'save_train_pred': False  # trainデータでの推論値を保存するか否か
    }

    # モデルのパラメータ
    # 回帰
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_round': 5000,
        'learning_rate': 0.0013,
        'max_depth': -2,
        'lambda_l1': 0.313,
        'lambda_l2': 1.056,
        'num_leaves': '43',
        'early_stopping_rounds': 1000,
        'verbose': 200,
        'random_state': 999
    }

    runner = Runner(run_name, ModelLGB, features, setting, lgb_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

    use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

    # モデルのconfigをjsonで保存
    value_list = [features, use_feature_name, lgb_params, cv, setting]
    save_model_config(key_list, value_list, dir_name, run_name)

    # runner.visualize_corr() # 相関係数を可視化して保存
    if cv.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 推論
    else:
        runner.run_train_cv()  # 学習
        ModelLGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
        runner.run_predict_cv()  # 推論

    Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    # #  ######################################################
    # # 学習・推論 xgboost ###################################

    # run nameの設定
    run_name = 'xgb'
    run_name = run_name + suffix
    dir_name = MODEL_DIR_NAME + run_name + '/'

    # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # 諸々の設定
    setting = {
        'run_name': run_name,  # run名
        'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'target': 'SalePrice',  # 目的変数
        'calc_shap': False,  # shap値を計算するか否か
        'save_train_pred': False  # trainデータでの推論値を保存するか否か
    }

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'num_round': 5000,
        'early_stopping_rounds': 1000,
        'verbose': 200,
        'random_state': 999
    }

    runner = Runner(run_name, ModelXGB, features, setting, xgb_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

    use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

    # モデルのconfigをjsonで保存
    value_list = [features, use_feature_name, xgb_params, cv, setting]
    save_model_config(key_list, value_list, dir_name, run_name)

    # runner.visualize_corr() # 相関係数を可視化して保存
    if cv.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 推論
    else:
        runner.run_train_cv()  # 学習
        ModelXGB.calc_feature_importance(dir_name, run_name, use_feature_name)  # feature_importanceを計算
        runner.run_predict_cv()  # 推論

    Submission.create_submission(run_name, dir_name, setting.get('target'))  # submit作成

    #####################################################
    # 学習・推論 catboost ###################################

    # run nameの設定
    run_name = 'cb'
    run_name = run_name + suffix
    dir_name = MODEL_DIR_NAME + run_name + '/'

    # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # 諸々の設定
    setting = {
        'run_name': run_name,  # run名
        'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'target': 'SalePrice',  # 目的変数
        'calc_shap': False,  # shap値を計算するか否か
        'save_train_pred': False  # trainデータでの推論値を保存するか否か
    }

    cb_params = {
        'loss_function': 'RMSE',
        'iterations': 50000,
        'od_type': "Iter",
        'depth': 10,
        'colsample_bylevel': 0.5,
        'early_stopping_rounds': 300,
        'l2_leaf_reg': 18,
        'random_seed': 42,
        'verbose_eval': 300,
        'use_best_model': True
    }

    runner = Runner(run_name, ModelCB, features, setting, cb_params, cv, FEATURE_DIR_NAME, MODEL_DIR_NAME)

    use_feature_name = runner.get_feature_name()  # 今回の学習で使用する特徴量名を取得

    # モデルのconfigをjsonで保存
    value_list = [features, use_feature_name, cb_params, cv, setting]
    save_model_config(key_list, value_list, dir_name, run_name)

    # runner.visualize_corr() # 相関係数を可視化して保存
    if cv.get('method') == 'None':
        runner.run_train_all()  # 全データで学習
        runner.run_predict_all()  # 推論
    else:
        runner.run_train_cv()  # 学習
        runner.run_predict_cv()  # 推論

    Submission.create_submission(run_name, dir_name, setting.get('target'))

    ######################################################
    # 学習・推論 stackingCVR ###################################

    # run nameの設定
    run_name = 'stack'
    run_name = run_name + suffix
    dir_name = MODEL_DIR_NAME + run_name + '/'

    # exist_check(MODEL_DIR_NAME, run_name)  # 実行可否確認
    my_makedirs(dir_name)  # runディレクトリの作成。ここにlogなどが吐かれる

    # 諸々の設定
    setting = {
        'run_name': run_name,  # run名
        'feature_directory': FEATURE_DIR_NAME,  # 特徴量の読み込み先ディレクトリ
        'target': 'SalePrice',  # 目的変数
    }

    gbr_params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 4,
        'max_features': 'sqrt',
        'min_samples_leaf': 15,
        'min_samples_split': 10,
        'loss': 'huber',
        'random_state': 42
    }

    stack_params = {
        'lgb_params': lgb_params,
        'xgb_params': xgb_params,
        'gbr_params': gbr_params,
        # Random Forest
        'rf_params': {
            'n_jobs': -1,
            'n_estimators': 500,
            'warm_start': True,
            # 'max_features': 0.2,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'verbose': 0
        },
        # Extra Trees
        'et_params': {
            'n_jobs': -1,
            'n_estimators': 500,
            # 'max_features': 0.5,
            'max_depth': 8,
            'min_samples_leaf': 2,
            'verbose': 0
        },
        # AdaBoost
        'ada_params': {
            'n_estimators': 500,
            'learning_rate': 0.75
        },
        # Gradient Boosting
        'gb_params': {
            'n_estimators': 500,
            # 'max_features': 0.2,
            'max_depth': 5,
            'min_samples_leaf': 2,
            'verbose': 0
        },
        # Support Vector Classifier
        'svc_params': {
            'kernel': 'linear',
            'C': 0.025
        }
    }

    # モデルのconfigをjsonで保存
    value_list = [features, '', stack_params, cv, setting]
    save_model_config(key_list, value_list, dir_name, run_name)

    model = ModelStack(setting, stack_params)
    pred = model.train()

    print('finish predict')

    submission = pd.read_csv('../data/raw/sample_submission.csv')
    submission['SalePrice'] = np.expm1(pred)
    submission.to_csv(dir_name + f'{run_name}_submission.csv', index=False, columns=['Id', 'SalePrice'])
    print(f'{run_name} - end create submission')