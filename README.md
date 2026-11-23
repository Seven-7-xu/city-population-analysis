# 城市人口分析与预测

> 基于 CCF BDCI 2023 赛题数据，对 40 座匿名城市 2000—2022 年多维数据进行清洗、建模与预测，输出 2023 年常住人口预测结果。

---

## 1. 项目简介

本项目为人工智能数据服务课程期末大作业，完成「数据清洗 → 特征工程 → 回归建模 → 可视化 → 业务建议」的完整数据科学闭环。

**数据来源**：[CCF BDCI 2023 赛题 #993](https://www.datafountain.cn/competitions/993)（仅限课程学习，禁止外传与商用）

---

## 2. 组成员与分工

| 代号 | 姓名 | 主责模块 | 具体负责 |
|------|------|---------|---------|
| A | 陈泽宇 | 模块一：数据清洗 + EDA | `data_loader.py` / `preprocess.py` / `01_eda.ipynb` |
| B | 陈仁杰 | 模块二：建模 + 评估 | `features.py` / `train.py` / `evaluate.py` / `predict.py` |
| C | 林琨杰 | 模块三：可视化 | `visualize.py` / `outputs/figures/` |
| D | 陈昭伊 | 模块四：报告 + 统筹 | `docs/分析报告.md` / `README.md` / 答辩 PPT |

---

## 3. 环境与运行方式

### 环境要求
- Python 3.10+
- Windows / macOS / Linux

### 安装依赖
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 一键运行全流程
```bash
python scripts/run_all.py
```

### 单独生成并校验提交文件
```bash
python scripts/make_submission.py
```

---

## 4. 数据获取说明

1. 访问 [赛题页面](https://www.datafountain.cn/competitions/993)，注册并登录
2. 下载 A 榜全部 **7 份 Excel 文件**；`submission_sample.csv` 为可选项（用于校验提交列名）
3. 把 7 份 Excel 原样放入 `data/raw/` 目录；样例文件若下载到也一并放入
4. **严禁手工修改原始文件**，所有清洗动作写在 `src/preprocess.py` 中
5. **原始数据禁止上传公开仓库**（`.gitignore` 已忽略 `data/raw/`）

> 未提供 `submission_sample.csv` 时，流程仍可运行，只是跳过「列名/顺序与样例一致」的校验。

---

## 5. 主要结果

> 重建模目标为对数增长率 `ln(pop_t / pop_{t-1})`，评估时用上年真实人口还原为人口水平后计算指标。

| 模型 | 主要超参数 | MSE | RMSE | MAE | R² |
|------|-----------|-----|------|-----|-----|
| RandomForestRegressor | n_estimators=150, max_depth=8 | 1150.12 | 33.91 | 15.18 | 0.9916 |
| LinearRegression | - | 1233.79 | 35.13 | 16.74 | 0.9910 |

**最优模型**：RandomForestRegressor（验证集 2019—2021，MAPE 2.01%，平均偏差 −5.16 万人）

---

## 6. 目录结构

```
city-population-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml          # 路径、模型超参数等配置
├── data/
│   ├── raw/                 # 原始数据（只读，禁止改动，已 gitignore）
│   ├── interim/             # 中间数据（合并后、未处理缺失）
│   └── processed/           # 建模用数据（清洗完成）
├── notebooks/
│   └── 01_eda.ipynb         # 探索性分析
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # 数据读取与体检
│   ├── preprocess.py        # 脏数据修正、宽长转换、合并、缺失值处理
│   ├── features.py          # 特征工程
│   ├── train.py             # 模型训练
│   ├── evaluate.py          # 指标计算与模型对比
│   ├── predict.py           # 2023 年预测与提交文件生成
│   └── visualize.py         # 全部绘图逻辑
├── scripts/
│   ├── run_all.py           # 一键跑通全流程
│   └── make_submission.py   # 生成并校验提交文件
├── models/                  # 训练好的模型（已 gitignore）
├── outputs/
│   ├── figures/             # 图表 PNG
│   ├── metrics/             # 指标 CSV、数据体检表
│   └── submission.csv       # 最终预测结果
└── docs/
    ├── data_dictionary.md   # 数据字典
    ├── 分析报告.md           # 主报告
    └── trae_log.md          # AI 使用记录
```

---

**数据来源声明**：本项目数据来源于 CCF 大数据与计算智能大赛 2023 赛题「城市人口分析与预测」，仅用于课程教学，不得用于任何商业用途。
