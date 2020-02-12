# regression_pipeline

## 参考
- https://github.com/takapy0210/ml_pipeline/blob/master/data/raw/sample_submission.csv
- https://www.takapy.work/entry/2019/12/14/165119

## 実行
```
# 特徴量作成
$ python3 run_fe.py

# 実行
$ python3 train_run.py

# 特徴量確認
$ python3 show_all_features.py

# ハイパラメータサーチ(lgmb)
$ python3 optuna_lbgm.py

```
# kaggle_pipeline
