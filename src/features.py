"""
features.py - 特征工程
======================
按任务书 4.2.1 构造 3 个必做衍生特征与若干选做特征。

⚠️ 特征泄漏处理（报告必写）：
1. pop_diff / pop_growth_rate / pop_flow_ratio 的「当期值」都由标签
   常住人口算出，直接入模会造成标签泄漏，验证集指标虚高。
   因此这三个特征照常计算并写入数据（用于展示与报告），
   但建模时改用它们的「滞后一期」版本（*_lag1，即上一年取值）。
2. 年龄结构 0-14 / 15-64 / 65+ 三个绝对人口数之和 ≈ 常住人口，
   按任务书「方案 a」不作为特征；只保留占比类衍生特征。
"""

import numpy as np
import pandas as pd

LABEL = "常住人口（万人）"
HUKOU = "户籍人口（万人）"

# 建模目标：对数增长率 log(pop_t / pop_{t-1})
# 相比直接预测人口水平，增长率目标不需要对经济特征做趋势外推，
# 可以显著缓解「系统性高估」问题（水平模型曾高估 11.4%，实际中位增速仅 2.8%）。
GROWTH_TARGET = "pop_log_growth"
ANCHOR_COL = "prev_pop"

NON_FEATURE = {"city_id", "year", LABEL, GROWTH_TARGET, ANCHOR_COL}
LABEL_DERIVED = {"pop_diff", "pop_growth_rate", "pop_flow_ratio"}
AGE_ABS = {"0-14", "15-64", "65+"}


def add_required_features(df: pd.DataFrame,
                          pop_col: str = LABEL,
                          hukou_col: str = HUKOU) -> pd.DataFrame:
    """3 个必做衍生特征"""
    df = df.sort_values(["city_id", "year"]).reset_index(drop=True)
    prev_pop = df.groupby("city_id")[pop_col].shift(1)

    df["pop_diff"] = df[pop_col] - prev_pop
    df["pop_growth_rate"] = (df[pop_col] - prev_pop) / prev_pop
    df["pop_flow_ratio"] = (df[pop_col] - df[hukou_col]) / df[hukou_col]
    df = df.replace([np.inf, -np.inf], np.nan)

    print("  [FEATURE] pop_diff / pop_growth_rate / pop_flow_ratio")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """为规避标签泄漏，额外生成三个特征的滞后一期版本用于建模"""
    for col in sorted(LABEL_DERIVED):
        df[f"{col}_lag1"] = df.groupby("city_id")[col].shift(1)
    print("  [FEATURE] pop_diff_lag1 / pop_growth_rate_lag1 / pop_flow_ratio_lag1")
    return df


def add_optional_features(df: pd.DataFrame,
                          urb_col: str = "urbanizationRate",
                          wage_col: str = "wage") -> pd.DataFrame:
    """选做特征（冲击加分项）"""
    df["year_since_2000"] = (df["year"] - 2000).astype("float64")

    if urb_col in df.columns:
        df["urb_gap"] = df.groupby("city_id")[urb_col].diff()

    if wage_col in df.columns:
        prev_wage = df.groupby("city_id")[wage_col].shift(1)
        df["wage_growth"] = (df[wage_col] - prev_wage) / prev_wage

    age_cols = [c for c in ["0-14", "15-64", "65+"] if c in df.columns]
    if len(age_cols) == 3:
        total = df[age_cols].sum(axis=1)
        df["ratio_0_14"] = df[age_cols[0]] / total
        df["ratio_65plus"] = df[age_cols[2]] / total
        df["old_child_ratio"] = df["ratio_65plus"] / df["ratio_0_14"]

    df = df.replace([np.inf, -np.inf], np.nan)
    print("  [FEATURE] year_since_2000 / urb_gap / wage_growth / 年龄占比")
    return df


def add_growth_target(df: pd.DataFrame, pop_col: str = LABEL) -> pd.DataFrame:
    """
    生成建模目标与还原锚点：
    - pop_log_growth = log(pop_t / pop_{t-1})，对数增长率
    - prev_pop       = pop_{t-1}，用于把预测增长率还原成人口水平
    """
    prev = df.groupby("city_id")[pop_col].shift(1)
    df[ANCHOR_COL] = prev
    df[GROWTH_TARGET] = np.log(df[pop_col] / prev)
    df = df.replace([np.inf, -np.inf], np.nan)
    print("  [TARGET] pop_log_growth（对数增长率）+ prev_pop（还原锚点）")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """返回可入模的数值特征列（显式排除标签、标签派生当期值、年龄绝对数）"""
    excluded = NON_FEATURE | LABEL_DERIVED | AGE_ABS
    cols = [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  [FEATURE] 入模特征 {len(cols)} 个：{cols}")
    return cols
