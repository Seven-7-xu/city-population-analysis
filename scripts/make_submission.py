"""
make_submission.py - 单独生成并校验提交文件
============================================
前提：已运行过 python scripts/run_all.py（存在 model_data.csv 与 final_model.pkl）
用法：python scripts/make_submission.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import joblib
import pandas as pd

from src.data_loader import load_config
from src.features import get_feature_cols
from src.predict import predict_2023, build_submission, validate_submission, save_submission


def main():
    cfg = load_config()

    data_path = os.path.join(cfg["paths"]["processed_data"], "model_data.csv")
    model_path = os.path.join(cfg["paths"]["models_dir"], "final_model.pkl")

    for path in (data_path, model_path):
        if not os.path.exists(path):
            print(f"❌ 缺少 {path}\n请先运行：python scripts/run_all.py")
            return

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    feature_cols = get_feature_cols(df)

    pred = predict_2023(df, feature_cols, model, cfg)
    submission = build_submission(pred, cfg)

    sample_path = os.path.join(cfg["paths"]["raw_data"], cfg["raw_files"]["submission_sample"])
    if validate_submission(submission, sample_path):
        path = save_submission(submission, cfg)
        print(f"\n预览（前 5 行）：\n{submission.head().to_string(index=False)}")
        print(f"\n完整文件：{path}")


if __name__ == "__main__":
    main()
