"""
visualize.py - 全部绘图逻辑（对应任务书 4.3）
============================================
图 1  2021 年各城市常住人口排名（前 15）        柱状图（必做）
图 2  主要城市常住人口随时间变化趋势            折线图（必做）
图 3  关键特征相关系数矩阵                      热力图（必做）
图 4  各模型验证集 RMSE 对比                    柱状图（必做）
图 5  2023 年 40 城预测人口排序                 横向条形图（必做）
图 6  最优模型特征重要性 Top 10                 条形图（选做 +3）
图 7  预测值 vs 真实值                          散点图（选做 +3）

规范：中文标题、轴名与单位、300 dpi、bbox_inches="tight"、元素不重叠。
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABEL = "常住人口（万人）"


def setup_style(font: str = "Microsoft YaHei") -> None:
    """中文字体与负号设置；seaborn 主题会覆盖字体，必须在之后重设"""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = [font, "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig, filename: str, cfg: dict) -> str:
    path = os.path.join(cfg["paths"]["figures_dir"], filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=cfg["figures"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {path}")
    return path


def _pick_col(df: pd.DataFrame, keyword: str) -> str | None:
    """按关键词找列名（避免写死中文全称）"""
    for c in df.columns:
        if keyword in str(c):
            return c
    return None


def fig1_ranking(df: pd.DataFrame, cfg: dict) -> str:
    """图 1：2021 年各城市常住人口排名前 15"""
    data = df[df["year"] == 2021].nlargest(15, LABEL)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(data["city_id"], data[LABEL], color="#4C72B0")
    ax.set_title("图1  2021 年各城市常住人口规模排名（前 15 名）", fontsize=13)
    ax.set_xlabel("城市")
    ax.set_ylabel("常住人口（万人）")
    ax.tick_params(axis="x", rotation=45)
    for bar, v in zip(bars, data[LABEL]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    return _save(fig, "fig1_population_ranking_2021.png", cfg)


def fig2_trend(df: pd.DataFrame, cfg: dict) -> str:
    """图 2：人口规模前 5 城市的常住人口时间趋势"""
    top5 = df[df["year"] == 2021].nlargest(5, LABEL)["city_id"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for city in top5:
        g = df[(df["city_id"] == city) & (df["year"] >= 2001)].sort_values("year")
        ax.plot(g["year"], g[LABEL], marker="o", markersize=3, linewidth=1.6, label=city)
    ax.set_title("图2  主要城市常住人口随时间变化趋势（2001—2021）", fontsize=13)
    ax.set_xlabel("年份")
    ax.set_ylabel("常住人口（万人）")
    ax.legend(title="城市", ncol=5, fontsize=9)
    return _save(fig, "fig2_population_trend.png", cfg)


def fig3_correlation(df: pd.DataFrame, cfg: dict) -> str:
    """图 3：关键特征相关系数矩阵热力图"""
    candidates = [LABEL, "户籍人口（万人）", "urbanizationRate", "wage",
                  "unemploymentRate", "employeesNumber", "pop_flow_ratio"]
    density = _pick_col(df, "人口密度")
    if density:
        candidates.append(density)
    cols = [c for c in candidates if c in df.columns]

    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"label": "相关系数"})
    ax.set_title("图3  关键特征相关系数矩阵", fontsize=13)
    ax.set_xlabel("特征")
    ax.set_ylabel("特征")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _save(fig, "fig3_correlation_heatmap.png", cfg)


def fig4_model_comparison(metrics: pd.DataFrame, cfg: dict) -> str:
    """图 4：各模型验证集 RMSE 对比"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(metrics["model"], metrics["RMSE"], color=["#4C72B0", "#DD8452"][:len(metrics)])
    ax.set_title("图4  各模型在验证集上的性能对比（RMSE 越小越好）", fontsize=12)
    ax.set_xlabel("模型")
    ax.set_ylabel("RMSE（万人）")
    for bar, v in zip(bars, metrics["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    return _save(fig, "fig4_model_rmse_comparison.png", cfg)


def fig5_prediction_ranking(submission: pd.DataFrame, cfg: dict) -> str:
    """图 5：2023 年 40 城预测人口排序（横向条形图）"""
    data = submission.sort_values("count")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(data["city_id"], data["count"], color="#55A868")
    ax.set_title("图5  2023 年 40 座城市常住人口预测排序", fontsize=13)
    ax.set_xlabel("预测常住人口（万人）")
    ax.set_ylabel("城市")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, "fig5_prediction_2023.png", cfg)


def fig6_feature_importance(importance: pd.DataFrame, cfg: dict, top_k: int = 10) -> str:
    """图 6（选做）：最优模型特征重要性 Top 10"""
    data = importance.head(top_k).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["feature"], data["importance"], color="#8172B3")
    ax.set_title(f"图6  最优模型特征重要性 Top {top_k}", fontsize=13)
    ax.set_xlabel("重要性")
    ax.set_ylabel("特征")
    return _save(fig, "fig6_feature_importance.png", cfg)


def fig7_pred_vs_true(y_true, y_pred, cfg: dict) -> str:
    """图 7（选做）：验证集预测值 vs 真实值，含 y = x 参考线"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=28, alpha=0.75, color="#4C72B0", label="样本点")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("图7  验证集预测值 vs 真实值", fontsize=13)
    ax.set_xlabel("真实常住人口（万人）")
    ax.set_ylabel("预测常住人口（万人）")
    ax.legend()
    return _save(fig, "fig7_pred_vs_true.png", cfg)


def generate_all_figures(df: pd.DataFrame, metrics: pd.DataFrame,
                         submission: pd.DataFrame, importance: pd.DataFrame,
                         y_val, y_pred, cfg: dict) -> dict:
    """一键生成全部 7 张图"""
    setup_style(cfg["figures"]["font_family"])
    paths = {
        "fig1": fig1_ranking(df, cfg),
        "fig2": fig2_trend(df, cfg),
        "fig3": fig3_correlation(df, cfg),
        "fig4": fig4_model_comparison(metrics, cfg),
        "fig5": fig5_prediction_ranking(submission, cfg),
        "fig6": fig6_feature_importance(importance, cfg),
        "fig7": fig7_pred_vs_true(y_val, y_pred, cfg),
    }
    print(f"  [FIG] 共生成 {len(paths)} 张图（必做 5 张 + 选做 2 张）")
    return paths
