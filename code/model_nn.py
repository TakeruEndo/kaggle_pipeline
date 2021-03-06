import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
# from keras.utils import np_utils
from keras import optimizers
# from keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from model import Model
from util import Util

import matplotlib.pyplot as plt

# 各foldのモデルを保存する配列
model_array = []


class ModelNN(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        scaler = StandardScaler()
        tr_x = scaler.fit_transform(tr_x)
        va_x = scaler.transform(va_x)

        # tr_x = tr_x.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        # va_x = va_x.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        # データのセット・スケーリング
        # validation = va_x is not None

        # パラメータ
        dropout = self.params['dropout']
        units = self.params['units']
        nb_epoch = self.params['nb_epoch']
        patience = self.params['patience']

        # ニューラルネットモデルの構築
        model = Sequential()
        model.add(Dense(units, activation='relu', input_shape=(tr_x.shape[1],)))
        model.add(Dropout(dropout))
        model.add(Dense(units))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        # The Output Layer :
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss="mean_squared_error", metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        save_best = ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)
        history = model.fit(
            tr_x, tr_y, epochs=nb_epoch, batch_size=128, verbose=1, validation_data=(va_x, va_y), callbacks=[save_best, early_stopping]
        )
            
        model_array.append(self.model)
            
        # モデル・スケーラーの保持
        model.load_weights('nn_model.w8')
        self.model = model
        self.scaler = scaler
        self.history = history

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict(te_x)
        return np.ravel(pred)  # 1次元に変換する

    def save_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(path, f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, path):
        model_path = os.path.join(path, f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(path, f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)

    def load_history(self):
        return self.history
