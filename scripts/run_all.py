"""
run_all.py - 一键跑通全流程
===========================
用法（在任意目录均可）：
    python scripts/run_all.py
"""

import os
import sys

# 统一以项目根目录为工作目录，避免在 VSCode 里直接运行时找不到 config/、outputs/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import joblib
import numpy as np

from src.data_loader import load_config
from src.preprocess import run_preprocessing
from src.features import (add_required_features, add_lag_features,
                          add_optional_features, add_growth_target,
                          get_feature_cols, GROWTH_TARGET, ANCHOR_COL)
from src.train import split_by_time, train_models, save_models
from src.evaluate import (evaluate_growth_models, get_best_model, feature_importance)
from src.predict import predict_2023, build_submission, validate_submission, save_submission
from src.visualize import generate_all_figures

LABEL = "常住人口（万人）"


def main():
    cfg = load_config()

    print("=" * 60)
    print("城市人口分析与预测 - 全流程开始")
    print("=" * 60)

    # ---------- 1. 清洗 ----------
    df = run_preprocessing(cfg)

    # ---------- 2. 特征工程 ----------
    print("\n" + "=" * 60)
    print("STEP 4/8 特征工程")
    print("=" * 60)
    df = add_required_features(df)
    df = add_lag_features(df)
    df = add_optional_features(df)
    df = add_growth_target(df)

    model_data_path = os.path.join(cfg["paths"]["processed_data"], "model_data.csv")
    df.to_csv(model_data_path, index=False, encoding="utf-8-sig")
    print(f"  [SAVE] 建模数据 -> {model_data_path}")

    feature_cols = get_feature_cols(df)
    df_model = df.dropna(subset=feature_cols + [GROWTH_TARGET, ANCHOR_COL]).reset_index(drop=True)
    print(f"  [DATA] 可建模样本 {len(df_model)} 条"
          f"（剔除首年滞后/增长率缺失 {len(df) - len(df_model)} 条）")

    # ---------- 3. 训练 ----------
    print("\n" + "=" * 60)
    print("STEP 5/8 按时间切分 + 模型训练（目标：对数增长率）")
    print("=" * 60)
    X, y = df_model[feature_cols], df_model[GROWTH_TARGET]
    train_mask, val_mask = split_by_time(df_model, cfg)
    models = train_models(X[train_mask], y[train_mask], cfg)

    # ---------- 4. 评估 ----------
    print("\n" + "=" * 60)
    print("STEP 6/8 模型评估（增长率还原为人口水平后计算指标）")
    print("=" * 60)
    metrics_path = os.path.join(cfg["paths"]["metrics_dir"], "metrics.csv")
    y_true_level = df_model.loc[val_mask, LABEL]
    metrics = evaluate_growth_models(models, X[val_mask],
                                     df_model.loc[val_mask, ANCHOR_COL],
                                     y_true_level, metrics_path)
    best_name, best_model = get_best_model(models, metrics)

    save_models(models, cfg["paths"]["models_dir"])
    final_path = os.path.join(cfg["paths"]["models_dir"], "final_model.pkl")
    joblib.dump(best_model, final_path)
    print(f"  [SAVE] 最优模型 -> {final_path}（{best_name}）")

    importance = feature_importance(best_model, feature_cols, X[train_mask], top_k=10)
    importance.to_csv(os.path.join(cfg["paths"]["metrics_dir"], "feature_importance.csv"),
                      index=False, encoding="utf-8-sig")

    # ---------- 5. 预测 ----------
    print("\n" + "=" * 60)
    print("STEP 7/8 2023 年预测与提交文件")
    print("=" * 60)
    pred = predict_2023(df_model, feature_cols, best_model, cfg)
    submission = build_submission(pred, cfg)
    sample_path = os.path.join(cfg["paths"]["raw_data"], cfg["raw_files"]["submission_sample"])
    if validate_submission(submission, sample_path):
        save_submission(submission, cfg)

    # ---------- 6. 可视化 ----------
    print("\n" + "=" * 60)
    print("STEP 8/8 生成图表")
    print("=" * 60)
    y_pred = df_model.loc[val_mask, ANCHOR_COL].values * np.exp(best_model.predict(X[val_mask]))
    generate_all_figures(df_model, metrics, submission, importance,
                         y_true_level, y_pred, cfg)

    # ---------- 汇总 ----------
    best_row = metrics[metrics["model"] == best_name].iloc[0]
    print("\n" + "=" * 60)
    print("全流程完成 ✅")
    print("=" * 60)
    print(f"  样本数          : {len(df_model)}")
    print(f"  入模特征数      : {len(feature_cols)}")
    print(f"  最优模型        : {best_name}"
          f"（RMSE={best_row['RMSE']:.2f} 万人，R²={best_row['R2']:.4f}）")
    print(f"  预测结果        : {os.path.join(cfg['paths']['output_dir'], 'submission.csv')}")
    print(f"  图表            : {cfg['paths']['figures_dir']}/（7 张）")
    print(f"  指标表          : {metrics_path}")


if __name__ == "__main__":
    main()
