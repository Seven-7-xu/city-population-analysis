"""
preprocess.py - 数据清洗完整 Pipeline
=====================================
职责：
1. 脏数据修正：ctiy31/ctiy22 拼写、年份带"年"字、年份浮点
2. 宽表转长表：工资水平、生活水平（6 Sheet）
3. 列名统一：城市 → city_id，年份 → year
4. 多表合并：以「人口规模」为主表，左连接其余表
5. 缺失值分层处理：删标签缺失 → 按城市插值 → 中位数兜底

对外唯一入口：run_preprocessing()
"""

import os
import re
import numpy as np
import pandas as pd
import yaml


CITY_NAME_FIX = {"ctiy31": "city31", "ctiy22": "city22"}


# ============================================================
# 基础工具
# ============================================================

def _fix_city_names(series: pd.Series) -> pd.Series:
    """修正城市名拼写错误并去掉首尾空格"""
    return series.astype(str).str.strip().replace(CITY_NAME_FIX)


def _year_to_int(series: pd.Series) -> pd.Series:
    """'2001年' / 2011.0 → Int64"""
    s = series.astype(str).str.replace("年", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _standardize(df: pd.DataFrame, city_col: str, year_col: str) -> pd.DataFrame:
    """统一列名并规范城市/年份"""
    out = df.rename(columns={city_col: "city_id", year_col: "year"}).copy()
    out["city_id"] = _fix_city_names(out["city_id"])
    out["year"] = _year_to_int(out["year"])
    out = out[out["city_id"].str.startswith("city")].copy()
    return out


def _norm_sheet(name: str) -> str:
    """去掉顿号/逗号/空格，兼容「第一、二、三产业就业人数」这类写法差异"""
    return re.sub(r"[、,，\s]", "", str(name))


def _resolve_sheet(fpath: str, wanted: str) -> str:
    """按表名精确/模糊匹配真实工作表名，避免因标点差异读不到数据"""
    with pd.ExcelFile(fpath) as xls:
        names = xls.sheet_names
    for n in names:
        if _norm_sheet(n) == _norm_sheet(wanted):
            return n
    for n in names:
        if _norm_sheet(wanted) in _norm_sheet(n) or _norm_sheet(n) in _norm_sheet(wanted):
            return n
    raise KeyError(f"{os.path.basename(fpath)} 中找不到工作表「{wanted}」，实际为 {names}")


def _read(raw_dir: str, filename: str, sheet_name=0, header=0) -> pd.DataFrame:
    fpath = os.path.join(raw_dir, filename)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"缺少原始数据文件：{fpath}\n请先把 7 份 Excel 放入 {raw_dir}/ 目录"
        )
    if isinstance(sheet_name, str):
        sheet_name = _resolve_sheet(fpath, sheet_name)
    return pd.read_excel(fpath, sheet_name=sheet_name, header=header)


# ============================================================
# 各表读取（长表）
# ============================================================

def _load_population(raw_dir, cfg) -> pd.DataFrame:
    """人口规模.xlsx —— 主表，含标签「常住人口（万人）」"""
    df = _read(raw_dir, cfg["raw_files"]["population_scale"])
    df = _standardize(df, cfg["columns"]["pop_city"], cfg["columns"]["pop_year"])
    for col in (cfg["columns"]["label"], cfg["columns"]["hukou"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  [人口规模] shape={df.shape}")
    return df


def _load_urbanization(raw_dir, cfg) -> pd.DataFrame:
    df = _read(raw_dir, cfg["raw_files"]["urbanization_rate"])
    df = _standardize(df, "城市名称", "年份")
    df["urbanizationRate"] = pd.to_numeric(df["urbanizationRate"], errors="coerce")
    print(f"  [城镇化率] shape={df.shape}")
    return df


def _load_density(raw_dir, cfg) -> pd.DataFrame:
    """注意：人口密度表的列顺序是 [年份, 城市名称, 数值]"""
    df = _read(raw_dir, cfg["raw_files"]["population_density"])
    df = _standardize(df, "城市名称", "年份")
    print(f"  [人口密度] shape={df.shape}")
    return df


def _load_age(raw_dir, cfg) -> pd.DataFrame:
    """年龄结构.xlsx —— 城市列名为「城市」，只有 2000/2010/2020 三个普查年"""
    df = _read(raw_dir, cfg["raw_files"]["age_structure"])
    df = _standardize(df, cfg["columns"]["age_city"], "年份")
    print(f"  [年龄结构] shape={df.shape}（仅 2000/2010/2020 三个真实观测年）")
    return df


def _load_employment(raw_dir, cfg) -> list[pd.DataFrame]:
    """就业信息.xlsx —— 3 个 Sheet，逐个标准化"""
    frames = []
    for sn in cfg["sheets"]["employment_info"]:
        df = _read(raw_dir, cfg["raw_files"]["employment_info"], sheet_name=sn)
        df = _standardize(df, "城市名称", "年份")
        frames.append(df)
        print(f"  [就业/{sn}] shape={df.shape}")
    return frames


def _load_wage(raw_dir, cfg) -> pd.DataFrame:
    """工资水平.xlsx —— 宽表转长表；年份列形如 '2001年'"""
    df = _read(raw_dir, cfg["raw_files"]["wage_level"], sheet_name="职工平均工资")
    year_col = df.columns[0]                   # averageWage
    city_cols = [c for c in df.columns if str(c).startswith("city")]
    long_df = df.melt(id_vars=[year_col], value_vars=city_cols,
                      var_name="city_id", value_name="wage")
    long_df = long_df.rename(columns={year_col: "year"})
    long_df["city_id"] = _fix_city_names(long_df["city_id"])
    long_df["year"] = _year_to_int(long_df["year"])
    long_df["wage"] = pd.to_numeric(long_df["wage"], errors="coerce")
    print(f"  [工资水平] 宽表→长表 shape={long_df.shape}")
    return long_df


def _load_living_standard(raw_dir, cfg) -> list[pd.DataFrame]:
    """
    生活水平.xlsx —— 6 个宽表 Sheet。
    真实文件的表头位置并不完全统一（首行可能是英文指标名），
    因此自动探测表头：找到第一行「至少有 2 个像年份的单元格」再按该行读取。
    """
    fpath = os.path.join(raw_dir, cfg["raw_files"]["living_standard"])
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"缺少原始数据文件：{fpath}\n请先把 7 份 Excel 放入 {raw_dir}/ 目录"
        )

    frames = []
    for sn in cfg["sheets"]["living_standard"]:
        sheet = _resolve_sheet(fpath, sn)
        probe = pd.read_excel(fpath, sheet_name=sheet, header=None)

        header_idx = 0
        for i in range(min(5, len(probe))):
            years = pd.to_numeric(
                probe.iloc[i].astype(str).str.replace("年", "", regex=False),
                errors="coerce",
            )
            if (years.notna() & years.between(1990, 2035)).sum() >= 2:
                header_idx = i
                break

        df = pd.read_excel(fpath, sheet_name=sheet, header=header_idx)
        city_col = df.columns[0]

        # 只保留「年份型」列，避免把备注列也 melt 进来
        year_cols = []
        for c in df.columns[1:]:
            v = str(c).replace("年", "").strip()
            if v.replace(".", "", 1).isdigit() and 1990 <= float(v) <= 2035:
                year_cols.append(c)

        df = df[df[city_col].astype(str).str.strip().str.startswith("city")].copy()
        long_df = df.melt(id_vars=[city_col], value_vars=year_cols,
                          var_name="year", value_name=sn)
        long_df = long_df.rename(columns={city_col: "city_id"})
        long_df["city_id"] = _fix_city_names(long_df["city_id"])
        long_df["year"] = _year_to_int(long_df["year"])
        long_df[sn] = pd.to_numeric(long_df[sn], errors="coerce")
        frames.append(long_df)
        print(f"  [生活/{sn}] 表头行={header_idx}，宽表→长表 shape={long_df.shape}")
    return frames


# ============================================================
# 合并
# ============================================================

def merge_tables(main_df: pd.DataFrame, others: list[pd.DataFrame]) -> pd.DataFrame:
    """以主表为准，按 city_id + year 左连接，并防止笛卡尔积导致行数膨胀"""
    merged = main_df.copy()
    base_rows = len(merged)

    for i, other in enumerate(others):
        dup = other.duplicated(subset=["city_id", "year"])
        if dup.any():
            print(f"  [WARN] 第 {i + 1} 张附表存在 {int(dup.sum())} 条重复键，已去重")
            other = other.drop_duplicates(subset=["city_id", "year"])
        merged = merged.merge(other, on=["city_id", "year"], how="left")

    if len(merged) != base_rows:
        raise ValueError(
            f"合并后行数异常：{base_rows} → {len(merged)}，请检查附表是否存在重复键"
        )
    print(f"  [MERGE] 主表 {base_rows} 行，合并 {len(others)} 张附表后仍为 {len(merged)} 行 ✅")
    return merged


# ============================================================
# 缺失值分层处理
# ============================================================

def handle_missing(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    分层策略（报告中需逐条说明理由）：
    1. 删除标签缺失的样本（不参与训练）
    2. 数值列按城市分组线性插值（时序平滑指标）
    3. inf/-inf → NaN
    4. 仍缺失的中位数兜底
    """
    df = df.copy()
    print(f"\n[缺失值] 处理前：{len(df)} 行，缺失单元格 {int(df.isna().sum().sum())}")

    # 0. 人口类字段不可能为 0：真实数据中 city28 的 2001—2005 年常住人口被记为 0
    #    （同期户籍人口 700 余万），属于「缺失被写成 0」，必须先还原为缺失
    for col in (label_col, "户籍人口（万人）"):
        if col in df.columns:
            n_zero = int((df[col] <= 0).sum())
            if n_zero:
                df.loc[df[col] <= 0, col] = np.nan
                print(f"  [修正] {col} 中 {n_zero} 个非正数值按缺失处理")

    # 1. 删除标签缺失
    before = len(df)
    df = df.dropna(subset=[label_col])
    print(f"  [删除] 标签缺失 {before - len(df)} 行 → 剩余 {len(df)} 行")

    # 2. 数值列按城市线性插值（先转 float，避免 Int64 无法插值）
    df = df.sort_values(["city_id", "year"]).reset_index(drop=True)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "year"]
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].astype("float64")
            df[col] = df.groupby("city_id")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )

    # 3. inf → NaN，避免比率计算产生无穷值
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4. 中位数兜底
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    remain = int(df.isna().sum().sum())
    print(f"  [完成] 处理后缺失单元格：{remain}")
    return df


# ============================================================
# 入口
# ============================================================

def run_preprocessing(cfg: dict | None = None, root: str = ".") -> pd.DataFrame:
    """完整预处理：读取 → 清洗 → 合并 → 缺失值处理 → 保存"""
    if cfg is None:
        with open(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
                  "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    raw_dir = os.path.join(root, cfg["paths"]["raw_data"])
    label_col = cfg["columns"]["label"]

    print("=" * 60)
    print("STEP 1/3 读取并清洗各表")
    print("=" * 60)
    pop = _load_population(raw_dir, cfg)
    others = [
        _load_urbanization(raw_dir, cfg),
        _load_density(raw_dir, cfg),
        _load_age(raw_dir, cfg),
        _load_wage(raw_dir, cfg),
        *_load_employment(raw_dir, cfg),
        *_load_living_standard(raw_dir, cfg),
    ]

    print("\n" + "=" * 60)
    print("STEP 2/3 多表合并（主键 city_id + year）")
    print("=" * 60)
    merged = merge_tables(pop, others)

    interim_path = os.path.join(root, cfg["paths"]["interim_data"], "merged_raw.csv")
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    merged.to_csv(interim_path, index=False, encoding="utf-8-sig")
    print(f"  [SAVE] 合并中间结果 -> {interim_path}")

    print("\n" + "=" * 60)
    print("STEP 3/3 缺失值分层处理")
    print("=" * 60)
    cleaned = handle_missing(merged, label_col)

    out_path = os.path.join(root, cfg["paths"]["processed_data"], "cleaned_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 清洗后数据 -> {out_path}  shape={cleaned.shape}")

    return cleaned


if __name__ == "__main__":
    df = run_preprocessing()
    print(df.head())
