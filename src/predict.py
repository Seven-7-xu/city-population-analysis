"""
predict.py - 2023 年预测与提交文件生成
=====================================
当前方案：增长率两步法
- 模型训练目标是对数增长率 ln(pop_t / pop_{t−1})，而不是人口水平；
- 用各城市近三年特征的平均变化量外推 2022、2023 年特征，
  分别预测 ĝ_2022、ĝ_2023；
- 按 pop_2023 = pop_2021 × exp(ĝ_2022 + ĝ_2023) 递推还原；
- 单年增长率约束 ±6%，结果相对 2021 年再设 ±30% 兜底。

提交格式必须与 submission_sample.csv 一致：city_id, year, count
"""

import os
import numpy as np
import pandas as pd

LABEL = "常住人口（万人）"


def extrapolate_features(df: pd.DataFrame, feature_cols: list[str],
                         target_year: int, window: int = 3) -> pd.DataFrame:
    """
    对每个城市，用最近 window 年的平均年度变化量，把特征外推到 target_year。
    这是方案 A 的核心：只外推「特征」，标签由模型预测。
    """
    rows = []
    for city, g in df.groupby("city_id", sort=False):
        g = g.sort_values("year")
        steps = target_year - int(g["year"].max())
        row = {"city_id": city, "year": target_year}
        for col in feature_cols:
            s = pd.to_numeric(g[col], errors="coerce").dropna().tail(window)
            if len(s) >= 2:
                avg_change = (s.iloc[-1] - s.iloc[0]) / (len(s) - 1)
                row[col] = s.iloc[-1] + avg_change * steps
            elif len(s) == 1:
                row[col] = s.iloc[0]
            else:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fill_with_median(X: pd.DataFrame) -> pd.DataFrame:
    """个别城市若某特征全缺失，用全体中位数兜底，保证模型可预测"""
    return X.fillna(X.median(numeric_only=True))


def _last_observed_pop(df: pd.DataFrame) -> pd.Series:
    """每个城市最后一年（2021）的真实常住人口，用于结果合理性约束"""
    last = df.sort_values("year").groupby("city_id").tail(1)
    return last.set_index("city_id")[LABEL]


def _sanitize(pred, base, bound: float):
    """
    结果合理性约束（防止线性外推在个别城市失控）：
    预测值必须为有限正数，且相对最近真实值不超过 ±bound。
    """
    pred = np.asarray(pred, dtype=float)
    base = np.asarray(base, dtype=float)
    pred = np.where(np.isfinite(pred), pred, base)
    return np.clip(pred, base * (1 - bound), base * (1 + bound))


def predict_growth(df: pd.DataFrame, feature_cols: list[str], model,
                   cfg: dict) -> pd.DataFrame:
    """
    增长率两步法（在方案 B 基础上改进）：
    1) 用外推特征分别预测 2022、2023 年的对数增长率
    2) pop_2023 = pop_2021 × exp(ĝ2022 + ĝ2023)
    增长率本身已包含趋势信息，无需再对人口水平做外推，可避免系统性高估。
    """
    p = cfg["prediction"]
    target_year = p["target_year"]
    mid_year = target_year - 1
    window = p["window_years"]
    bound = p.get("growth_bound", 0.06)

    X_mid = _fill_with_median(
        extrapolate_features(df, feature_cols, mid_year, window)[feature_cols])
    future = extrapolate_features(df, feature_cols, target_year, window)
    X_future = _fill_with_median(future[feature_cols])

    g_mid = np.clip(model.predict(X_mid), -bound, bound)
    g_future = np.clip(model.predict(X_future), -bound, bound)

    base = _last_observed_pop(df).reindex(future["city_id"]).values
    pred = base * np.exp(g_mid + g_future)
    pred = _sanitize(pred, base, p.get("one_step_bound", 0.30))

    print(f"  [PREDICT] 增长率两步法：{len(pred)} 座城市 → {target_year} 年")
    print(f"  [SANITY] 单年增长率约束 ±{bound:.0%}，"
          f"两年累计中位变化 {(np.median(np.exp(g_mid + g_future)) - 1):.2%}")
    return pd.DataFrame({"city_id": future["city_id"], "year": target_year, "count": pred})


def predict_2023(df: pd.DataFrame, feature_cols: list[str], model, cfg: dict) -> pd.DataFrame:
    """按配置选择预测方案（当前流程统一使用增长率两步法）"""
    method = cfg["prediction"]["method"]
    if method != "growth":
        raise ValueError(
            f"未知的预测方法：{method}。当前模型以对数增长率为目标训练，"
            f"只能使用 growth 方案；若要用人口水平外推，需另训一个水平目标模型"
        )
    return predict_growth(df, feature_cols, model, cfg)


def build_submission(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    按 submission_sample.csv 的字段与顺序输出：city_id, year, count
    并保证 city1—city40 全部覆盖、无空值、无负值。
    """
    target_year = cfg["prediction"]["target_year"]
    sub = pred[["city_id", "year", "count"]].copy()
    sub["year"] = target_year
    sub["count"] = sub["count"].round(2)

    all_cities = {f"city{i}" for i in range(1, 41)}
    missing = all_cities - set(sub["city_id"])
    if missing:
        raise ValueError(f"预测结果缺少城市：{sorted(missing)}")

    sub = sub.sort_values("city_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    sub = sub.reset_index(drop=True)
    return sub


def validate_submission(sub: pd.DataFrame, sample_path: str | None = None) -> bool:
    """提交前自检，全部通过才写出文件"""
    ok = True

    if len(sub) != 40:
        print(f"  ❌ 行数应为 40，实际 {len(sub)}")
        ok = False

    expected = [f"city{i}" for i in range(1, 41)]
    if sorted(sub["city_id"]) != sorted(expected):
        print("  ❌ 城市编号未完整覆盖 city1—city40")
        ok = False

    if sub["count"].isna().any():
        print("  ❌ 存在空值预测")
        ok = False
    if (sub["count"] <= 0).any():
        print("  ❌ 存在非正数预测")
        ok = False

    if sample_path and os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        if list(sample.columns) != list(sub.columns):
            print(f"  ❌ 列名/顺序与样例不一致：{list(sub.columns)} vs {list(sample.columns)}")
            ok = False

    print("  ✅ 提交文件校验通过" if ok else "  ❌ 提交文件校验失败")
    return ok


def save_submission(sub: pd.DataFrame, cfg: dict) -> str:
    path = os.path.join(cfg["paths"]["output_dir"], "submission.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [SAVE] 提交文件 -> {path}（{len(sub)} 行）")
    return path
