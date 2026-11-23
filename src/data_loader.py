"""
data_loader.py - 数据读取与体检
===============================
只负责「读」和「体检」，不做任何清洗（清洗在 preprocess.py）。
"""

import os
import yaml
import pandas as pd


def load_config(config_path: str | None = None) -> dict:
    """加载 config/config.yaml"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_excel(raw_dir: str, filename: str, sheet_name=0, header=0) -> pd.DataFrame:
    """读取单个 Excel；文件缺失时给出明确提示，而不是抛裸异常。"""
    fpath = os.path.join(raw_dir, filename)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"缺少原始数据文件：{fpath}\n"
            f"请先到 https://www.datafountain.cn/competitions/993 下载 A 榜数据，"
            f"原样放入 {raw_dir}/ 目录（共 7 份 Excel + submission_sample.csv）"
        )
    return pd.read_excel(fpath, sheet_name=sheet_name, header=header)


def load_raw_tables(cfg: dict, root: str = ".") -> dict:
    """
    按 config 读取全部 7 份原始 Excel。

    Returns
    -------
    dict
        单 Sheet 文件 → DataFrame；
        多 Sheet 文件（就业信息 / 生活水平）→ {sheet 名: DataFrame}
    """
    raw_dir = os.path.join(root, cfg["paths"]["raw_data"])
    files = cfg["raw_files"]
    sheets = cfg["sheets"]

    tables = {}
    for key in ["population_scale", "urbanization_rate",
                "population_density", "age_structure", "wage_level"]:
        tables[key] = read_excel(raw_dir, files[key])
        print(f"  [LOAD] {files[key]:<12} shape={tables[key].shape}")

    tables["employment_info"] = {}
    for sn in sheets["employment_info"]:
        tables["employment_info"][sn] = read_excel(raw_dir, files["employment_info"], sheet_name=sn)
        print(f"  [LOAD] {files['employment_info']} / {sn} shape={tables['employment_info'][sn].shape}")

    tables["living_standard"] = {}
    for sn in sheets["living_standard"]:
        tables["living_standard"][sn] = read_excel(
            raw_dir, files["living_standard"], sheet_name=sn
        )
        print(f"  [LOAD] {files['living_standard']} / {sn} shape={tables['living_standard'][sn].shape}")

    sample_path = os.path.join(raw_dir, files["submission_sample"])
    if os.path.exists(sample_path):
        tables["submission_sample"] = pd.read_csv(sample_path)
        print(f"  [LOAD] {files['submission_sample']:<12} shape={tables['submission_sample'].shape}")

    return tables


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def profile_tables(tables: dict, save_path: str | None = None) -> pd.DataFrame:
    """
    数据体检：逐表输出行数、列数、城市数、年份范围、缺失单元格数与缺失率。
    结果同时打印并保存为 CSV，可直接作为报告「数据说明」章节的素材。
    """
    rows = []
    for key, obj in tables.items():
        if isinstance(obj, dict):
            items = [(f"{key}/{sub}", df) for sub, df in obj.items()]
        else:
            items = [(key, obj)]

        for name, df in items:
            city_col = _find_col(df, ["城市名称", "城市", "city_id"])
            year_col = _find_col(df, ["年份", "year"])
            cities = df[city_col].dropna().nunique() if city_col else "-"
            if year_col is not None:
                years = pd.to_numeric(
                    df[year_col].astype(str).str.replace("年", "", regex=False),
                    errors="coerce",
                )
                year_range = f"{int(years.min())}—{int(years.max())}" if years.notna().any() else "-"
            else:
                year_range = "宽表（年份在列）"

            total_cells = df.shape[0] * df.shape[1]
            missing = int(df.isna().sum().sum())
            rows.append({
                "table": name,
                "rows": df.shape[0],
                "cols": df.shape[1],
                "cities": cities,
                "year_range": year_range,
                "missing_cells": missing,
                "missing_rate": round(missing / total_cells, 4) if total_cells else 0.0,
            })

    profile = pd.DataFrame(rows)
    print("\n" + profile.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        profile.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] 数据体检表 -> {save_path}")
    return profile


if __name__ == "__main__":
    cfg = load_config()
    tbl = load_raw_tables(cfg)
    profile_tables(tbl, os.path.join(cfg["paths"]["metrics_dir"], "data_profile.csv"))
