"""
train.py - 模型训练
===================
任务书 4.2.2 / 4.2.3：
- 按时间切分（≤2018 训练，2019—2021 验证），严禁随机打乱
- 必做两个模型：LinearRegression（基线）+ RandomForestRegressor
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def split_by_time(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """
    ⚠️ 按时间切分：同一城市不同年份不能同时出现在训练集与验证集。
    返回 (train_mask, val_mask)
    """
    sp = cfg["splitting"]
    train_mask = df["year"] <= sp["train_end_year"]
    val_mask = (df["year"] >= sp["val_start_year"]) & (df["year"] <= sp["val_end_year"])

    print(f"  [SPLIT] 训练集 {int(train_mask.sum())} 条（year ≤ {sp['train_end_year']}）")
    print(f"  [SPLIT] 验证集 {int(val_mask.sum())} 条（{sp['val_start_year']}—{sp['val_end_year']}）")
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("训练集或验证集为空，请检查 splitting 配置")
    return train_mask, val_mask


def train_models(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict) -> dict:
    """训练两个必做模型"""
    trained = {}

    lr = LinearRegression(**cfg["models"]["linear_regression"].get("params", {}))
    lr.fit(X_train, y_train)
    trained["LinearRegression"] = lr
    print("  [TRAIN] LinearRegression 完成")

    rf_params = cfg["models"]["random_forest"].get("params", {})
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    trained["RandomForestRegressor"] = rf
    print(f"  [TRAIN] RandomForestRegressor 完成 "
          f"(n_estimators={rf_params.get('n_estimators')}, max_depth={rf_params.get('max_depth')})")

    return trained


def save_models(models: dict, models_dir: str = "models") -> None:
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  [SAVE] {name} -> {path}")


def load_model(path: str):
    return joblib.load(path)
