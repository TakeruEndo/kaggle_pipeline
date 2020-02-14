データコンペ用のパイプライン

## 初めに
本パイプラインは[takapyさんのコード](https://github.com/takapy0210/ml_pipeline)をクローンして作成いたしました :bow: \
また、kaggleの[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)コンペを想定して作成してあります

## 推論
```
$ python3 train_run.py
```
`lightGBM`・`XGBoost`・`CatBoost`・`Stackingモデル`・`Neural Network`の推論が走る

## ハイパ-パラメータサーチ(lgmb)
```
$ python3 optuna_lbgm.py
```
optunaによるlightGBMのハイパーパラメータの探索を行う


## 特徴量作成
基本的にnotebookで特徴量生成を実施する
```
# 特徴量作成
$ python3 run_fe.py

# 特徴量確認
$ python3 show_all_features.py
```

## 参考
#### pipeline
- https://www.takapy.work/entry/2019/12/14/165119

#### アンサンブル
- https://qiita.com/hkthirano/items/2c35a81fbc95f0e4b7c1

#### catboost
- https://blog.amedama.jp/entry/catboost

#### optuna
- https://github.com/optuna/optuna

#### その他
- kaggleで勝つデータ分析の技術(2019)
- 各モデルのドキュメントやnotebook記載のカーネル

