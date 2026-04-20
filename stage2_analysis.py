# %% [markdown]
# # 第二阶段：统计分析
#
# 本 notebook 读取第一阶段输出的 CSV 数据文件，执行全部统计分析并生成图表。
#
# **输入文件**：
# - `results/rq1_rq2_responses.csv`
# - `results/rq3_responses.csv`
# - `results/human_spotcheck_sample.csv` (可选，需人工填写后)
#
# **输出**：
# - `results/figures/*.png` — 全部可视化图表 (10 张)
# - `results/analysis_summary.csv` — 关键统计量汇总表
# - 控制台输出全部假设检验结果
#
# **可反复运行**：不调用任何 API，只做本地计算，秒级完成。

# %%
# ====== Cell 0: 安装分析依赖 ======
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy", "pandas", "scipy", "matplotlib", "seaborn",
    "statsmodels", "scikit-posthocs", "scikit-learn", "openpyxl"])
print("分析依赖安装完成 ✓")

# %%
# ====== Cell 1: 设置环境 & 加载数据 ======
import os
from pathlib import Path

# ★ 如果 notebook 不在 persona_hallucination/ 文件夹内，取消下行注释：
# os.chdir("/你的路径/persona_hallucination")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import pandas as pd
import config

# 检查数据文件
for name, fname in [("RQ1/RQ2", "rq1_rq2_responses.csv"), ("RQ3", "rq3_responses.csv")]:
    p = config.RESULTS_DIR / fname
    if p.exists():
        n = len(pd.read_csv(p))
        print(f"✓ {name}: {n} 条记录")
    else:
        print(f"✗ {name}: 文件不存在 — 请先完成第一阶段")


# %% [markdown]
# ## 一键运行全部分析
# 下面的 Cell 会依次运行 H1-H7 的全部检验和可视化。
# 你也可以跳过这个 Cell，在后面逐项单独运行。

# %%
# ====== Cell 2: 一键运行全部 ======
from analysis import run_all_analyses
run_all_analyses()


# %% [markdown]
# ## 逐项分析（可选）
# 如果你想单独运行某个假设的分析，使用以下 Cell。

# %%
# ====== Cell 3-1: 单独运行 H1 (Persona 风格效应) ======
from analysis import load_rq12, analyze_h1
df12 = load_rq12()
analyze_h1(df12)

# %%
# ====== Cell 3-2: 单独运行 H2 (Persona → 幻觉率) ======
from analysis import analyze_h2
analyze_h2(df12)

# %%
# ====== Cell 3-3: 单独运行 H3 (Persona × Model 交互) ======
from analysis import analyze_h3
analyze_h3(df12)

# %%
# ====== Cell 3-4: 单独运行 H4 (剂量-反应关系) ======
from analysis import analyze_h4
analyze_h4(df12)

# %%
# ====== Cell 3-5: 单独运行 Bootstrap 排名稳定性 ======
from analysis import analyze_rank_stability
analyze_rank_stability(df12)

# %%
# ====== Cell 3-6: 单独运行 H5 (切换残留效应) ======
from analysis import load_rq3, analyze_h5
df3 = load_rq3()
analyze_h5(df3)

# %%
# ====== Cell 3-7: 单独运行 H6 (指数衰减拟合) ======
from analysis import analyze_h6
decay_results = analyze_h6(df3)
if len(decay_results) > 0:
    print("\n衰减拟合参数:")
    print(decay_results.to_string(index=False))

# %%
# ====== Cell 3-8: 单独运行 H7 (语义距离效应) ======
from analysis import analyze_h7
analyze_h7(df3)

# %%
# ====== Cell 3-9: Cohen's κ (人工验证) ======
# 前提: 你已在 human_spotcheck_sample.csv 中填写了 human_verdict 列
from analysis import analyze_cohens_kappa
analyze_cohens_kappa()


# %% [markdown]
# ## 查看生成的图表

# %%
# ====== Cell 4: 在 Notebook 内显示所有图表 ======
import glob
from IPython.display import display, Image

fig_dir = config.RESULTS_DIR / "figures"
fig_files = sorted(glob.glob(str(fig_dir / "*.png")))

print(f"共 {len(fig_files)} 张图表:\n")
for fp in fig_files:
    print(f"{'─'*50}")
    print(f"📊 {Path(fp).name}")
    display(Image(filename=fp, width=700))

# 图表用途对照:
# h1_certainty_score_boxplots.png  — H1: CS 分布 (按条件×模型)
# h1_cs_dose_response.png          — H1: CS 随 confidence 的变化
# h2_hr_heatmap.png                — H2: HR 热力图 (条件×模型)
# h2_ar_vs_hr_scatter.png          — H2: AR vs HR 散点 (suppression mechanism)
# h3_persona_model_interaction.png — H3: Persona×Model 交互图
# h4_hr_dose_response.png          — H4: HR 剂量-反应曲线
# h5_clean_vs_postswitch.png       — H5: Clean-start vs Post-switch HR 柱状图
# h6_rhe_decay.png                 — H6: RHE 衰减曲线 + 指数拟合
# h6_pps_decay.png                 — H6: PPS 衰减曲线
# rank_stability.png               — Bootstrap 排名稳定性分布


# %% [markdown]
# ## 导出汇总数据

# %%
# ====== Cell 5: 导出关键数据为 Excel (方便做报告) ======
from analysis import load_rq12, generate_summary_table, _group_hr_ci

df12 = load_rq12()
summary = generate_summary_table(df12)

# 额外导出: 按条件汇总的详细表
detail = _group_hr_ci(df12, ["condition_id", "persona_category", "confidence_level", "model_key"])
detail_path = config.RESULTS_DIR / "hr_detail_by_condition.csv"
detail.to_csv(detail_path, index=False)
print(f"详细 HR 表: {detail_path}")

# Excel 汇总 (多个 sheet)
try:
    xlsx_path = config.RESULTS_DIR / "analysis_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        detail.to_excel(writer, sheet_name="HR_Detail", index=False)
    print(f"Excel 汇总: {xlsx_path}")
except Exception as e:
    print(f"Excel 导出失败: {e} (CSV 仍可用)")


# %% [markdown]
# ## 自定义分析示例

# %%
# ====== Cell 6: 自定义分析模板 ======
# 你可以在这里用 CSV 做任何额外分析

import pandas as pd
import config

# 加载数据
df = pd.read_csv(config.RESULTS_DIR / "rq1_rq2_responses.csv")

# 示例 1: 按数据集分组查看 HR
print("=== 按数据集分组的 HR ===")
for ds in df["dataset"].unique():
    sub = df[df["dataset"]==ds]
    incorrect = (sub["judge_verdict"]=="incorrect").sum()
    evaluable = sub["judge_verdict"].isin(["correct","incorrect"]).sum()
    hr = incorrect / evaluable if evaluable > 0 else float("nan")
    print(f"  {ds}: HR={hr:.3f} (n={evaluable})")

# 示例 2: 任何你想做的 groupby / pivot / 统计...
# print(df.groupby(["model_key","persona_category"])["certainty_score"].describe())
