import pandas as pd
import sys
import os
import csv
import yaml
# from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from base import Feature, get_arguments, generate_features
import warnings

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(file)

RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
Feature.dir = yml['SETTING']['FEATURE_PATH']  # 生成した特徴量の出力場所
feature_memo_path = Feature.dir + '_features_memo.csv'


# Target
class mpg(Feature):
    def create_features(self):
        self.train['mpg'] = train['mpg']
        create_memo('mpg', '1ガロン当たりの走行距離。今回の目的変数。')


# 欠損値をNoneに置き換え
class PoolQC(Feature):
    def create_features(self):
        self.train['PoolQC'] = train['PoolQC'].fillna("None")
        self.test['PoolQC'] = test['PoolQC'].fillna("None")
        create_memo('PoolQC', 'プールの品質')


# 欠損値をNoneに置き換え
class MiscFeature(Feature):
    def create_features(self):
        self.train['MiscFeature'] = train['MiscFeature'].fillna("None")
        self.test['MiscFeature'] = test['MiscFeature'].fillna("None")
        create_memo('MiscFeature', '他のカテゴリに含まれていないその他の機能')


# 欠損値をNoneに置き換え
class Alley(Feature):
    def create_features(self):
        self.train['Alley'] = train['Alley'].fillna("None")
        self.test['Alley'] = test['Alley'].fillna("None")
        create_memo('Alley', '路地へのアクセスの種類')


# 欠損値をNoneに置き換え
class Fence(Feature):
    def create_features(self):
        self.train['Fence'] = train['Fence'].fillna("None")
        self.test['Fence'] = test['Fence'].fillna("None")
        create_memo('Fence', 'フェンスの品質')


class GarageType(Feature):
    def create_features(self):
        self.train['GarageType'] = train['GarageType'].fillna("None")
        self.test['GarageType'] = test['GarageType'].fillna("None")
        create_memo('GarageType', 'ガレージの場所')


class GarageFinish(Feature):
    def create_features(self):
        self.train['GarageFinish'] = train['GarageFinish'].fillna("None")
        self.test['GarageFinish'] = test['GarageFinish'].fillna("None")
        create_memo('GarageFinish', 'ガレージの内部仕上げ')


class GarageQual(Feature):
    def create_features(self):
        self.train['GarageQual'] = train['GarageQual'].fillna("None")
        self.test['GarageQual'] = test['GarageQual'].fillna("None")
        create_memo('GarageQual', 'ガレージの品質')


class GarageCond(Feature):
    def create_features(self):
        self.train['GarageCond'] = train['GarageCond'].fillna("None")
        self.test['GarageCond'] = test['GarageCond'].fillna("None")
        create_memo('GarageCond', 'ガレージの状態')


# 全体の平均値で埋める
class LotFrontage(Feature):
    def create_features(self):
        all_data = pd.concat((train, test)).reset_index(drop=True)
        self.train['LotFrontage'] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
        self.test['LotFrontage'] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
        create_memo('LotFrontage', 'プロパティに接続されている通りの直線フィート')


class BsmtFinSF1(Feature):
    def create_features(self):
        self.train['BsmtFinSF1'] = train['BsmtFinSF1'].fillna(0)
        self.test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
        create_memo('BsmtFinSF1', 'タイプ1完成した平方フィート')


class cylinders(Feature):
    def create_features(self):
        self.train['cylinders'] = train['cylinders']
        self.test['cylinders'] = test['cylinders']
        create_memo('cylinders', 'シリンダー')


class horsepower(Feature):
    def create_features(self):
        self.train['horsepower'] = train['horsepower']
        self.test['horsepower'] = test['horsepower']
        create_memo('horsepower', '馬力')


class weight(Feature):
    def create_features(self):
        self.train['weight'] = train['weight']
        self.test['weight'] = test['weight']
        create_memo('weight', '重量')


class acceleration(Feature):
    def create_features(self):
        self.train['acceleration'] = train['acceleration']
        self.test['acceleration'] = test['acceleration']
        create_memo('acceleration', '加速度')


class model_year(Feature):
    def create_features(self):
        self.train['model year'] = train['model year']
        self.test['model year'] = test['model year']
        create_memo('model year', '年式')


class origin(Feature):
    def create_features(self):
        self.train['origin'] = train['origin']
        self.test['origin'] = test['origin']
        create_memo('origin', '起源')


class car_label_encoder(Feature):
    def create_features(self):
        cols = 'car name'
        tmp_df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
        le = LabelEncoder().fit(tmp_df[cols])
        self.train['car name_label_encoder'] = le.transform(train[cols])
        self.test['car name_label_encoder'] = le.transform(test[cols])
        create_memo('car name_label_encoder', '車の名前ををラベルエンコーディングしたもの')


# https://github.com/OctopCat/SIGNATE_mynavi2019/blob/master/KERNEL_01.ipynb
class car_count(Feature):
    def create_features(self):
        self.train['car_count_all'] = train['car name'].map(pd.concat(
            [train['car name'], test['car name']], ignore_index=True).value_counts(dropna=False)
        )
        self.test['car_count_all'] = test['car name'].map(pd.concat(
            [train['car name'], test['car name']], ignore_index=True).value_counts(dropna=False)
        )


# 欠損値補完
# class age_mis_val_median(Feature):
#     def create_features(self):
#         self.train['Age_mis_val_median'] = train['Age'].fillna(train['Age'].median())
#         self.test['Age_mis_val_median'] = test['Age'].fillna(test['Age'].median())
#         create_memo('Age_mis_val_median','年齢の欠損値を中央値で補完したもの')


class displacement_plus_horsepower(Feature):
    def create_features(self):
        self.train['displacement_plus_horsepower'] = train['displacement'] + train['horsepower']
        self.test['displacement_plus_horsepower'] = test['displacement'] + test['horsepower']
        create_memo('Family_displacement_plus_horsepowerSize', '馬力＋排出量')


class power(Feature):
    def create_features(self):
        self.train['power'] = train['weight'] * train['acceleration']
        self.test['power'] = test['weight'] * test['acceleration']
        create_memo('power', '力')


# 特徴量メモcsvファイル作成
def create_memo(col_name, desc):

    file_path = Feature.dir + '/_features_memo.csv'
    if not os.path.isfile(file_path):
        with open(file_path, "w"):
            pass

    with open(file_path, 'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:
            return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])


if __name__ == '__main__':

    # CSVのヘッダーを書き込み
    create_memo('特徴量', 'メモ')

    args = get_arguments()
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)
