"""
evaluate.py - 模型评估
======================
任务书 4.2.4：MSE / RMSE / MAE / R²、模型对比表、特征重要性
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": round(mse, 4),
        "RMSE": round(float(np.sqrt(mse)), 4),   # 单位：万人
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_models(models: dict, X_val: pd.DataFrame, y_val: pd.Series,
                    save_path: str | None = None) -> pd.DataFrame:
    """多模型验证集评估，输出并保存对比表"""
    rows = []
    for name, model in models.items():
        pred = model.predict(X_val)
        metrics = compute_metrics(y_val, pred)
        metrics["model"] = name

        params = model.get_params()
        keys = {}
        for k in ("n_estimators", "max_depth"):
            if k in params:
                keys[k] = params[k]
        metrics["params"] = " ".join(f"{k}={v}" for k, v in keys.items()) or "-"
        rows.append(metrics)

    result = pd.DataFrame(rows)[["model", "params", "MSE", "RMSE", "MAE", "R2"]]
    print("\n[评估] 模型对比表（RMSE 单位：万人）")
    print(result.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 指标 -> {save_path}")
    return result


def evaluate_growth_models(models: dict, X_val: pd.DataFrame, anchor_pop,
                           y_true_level, save_path: str | None = None) -> pd.DataFrame:
    """
    增长率模型评估：先把预测的对数增长率还原为人口水平
    （pop_t = pop_{t-1} × exp(ĝ)），再计算指标，使结果可与其他方案直接比较。
    """
    rows = []
    anchor = np.asarray(anchor_pop, dtype=float)
    for name, model in models.items():
        growth = model.predict(X_val)
        pred = anchor * np.exp(growth)
        metrics = compute_metrics(y_true_level, pred)
        metrics["model"] = name

        params = model.get_params()
        keys = {k: params[k] for k in ("n_estimators", "max_depth") if k in params}
        metrics["params"] = " ".join(f"{k}={v}" for k, v in keys.items()) or "-"
        rows.append(metrics)

    result = pd.DataFrame(rows)[["model", "params", "MSE", "RMSE", "MAE", "R2"]]
    print("\n[评估] 增长率模型（已还原为人口水平，RMSE 单位：万人）")
    print(result.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 指标 -> {save_path}")
    return result


def get_best_model(models: dict, result: pd.DataFrame, metric: str = "RMSE"):
    ascending = metric != "R2"
    best_name = result.sort_values(metric, ascending=ascending).iloc[0]["model"]
    print(f"[BEST] 最优模型：{best_name}（按 {metric}）")
    return best_name, models[best_name]


def feature_importance(model, feature_cols: list[str], X: pd.DataFrame | None = None,
                       top_k: int = 10) -> pd.DataFrame:
    """
    树模型用 feature_importances_；
    线性模型用 |系数| × 特征标准差（标准化重要性），
    否则不同量纲的系数无法直接比较（例如占比 0.1 与人头数 1000）。
    """
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        if X is not None:
            values = coef * np.asarray(X.std(axis=0), dtype=float)
        else:
            values = coef
    else:
        return pd.DataFrame()

    imp = (pd.DataFrame({"feature": feature_cols, "importance": values})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    print(f"\n[特征重要性] Top {top_k}（线性模型已按特征标准差标准化）")
    print(imp.head(top_k).to_string(index=False))
    return imp
