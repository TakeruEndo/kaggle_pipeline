import numpy as np
import panas as pd
from scikit.metrics import log_loss
from sklearn.model_selection import KFold

# models.pyにMode1Xgb, Model1NN, Model2Linearを定義しているものとする
# 各クラスはfitで学習し、predictで予測値の学習をする

from models import Model1Xgb, Model1NN, Model2Linear


# 学習データに対する「目的変数をしらない」予測値と、テストデータに対す流予測値を返す関数
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
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


# 1層目のモデル
# pred_train_1a, pred_train_1bは、学習データのクロスバリデーションでの予測値
# pred_test_1a, pred_test_1bは、テストデータの予測値
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, train_x, train_y, test_x)

model_1b = Model1NN()
pred_train_1b, pred_test_1b = predict_cv(model_1b, train_x_nn, train_y, test_x_nn)

# 1層目のモデルの評価
print(f'logloss: (log_loss(train_y, pred_train_1a, eps=1e-7):.4f)')
print(f'logloss: (log_loss(train_y, pred_train_1b, eps=1e-7):.4f)')

# 予測値を特徴量としてデータフレームを作成
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})

# 2層目のモデル
# pred_train_2は、２層目のモデルの学習データのクロスバリデーションの予測値
# pred_test_2は、２層目のモデルのテストデータの予測値
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2, train_x_2, train_y, test_x_2)
print(f'logloss: {log_loss'train_y, pred_train_2, eps=1e-7}:.4f')
