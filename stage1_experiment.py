# %% [markdown]
# # 第一阶段：实验 + 指标计算
#
# 本 notebook 完成所有 API 调用、指标计算，最终输出数据文件：
# - `results/rq1_rq2_responses.csv` — RQ1/RQ2 全部响应 + 指标 + judge 判定
# - `results/rq3_responses.csv` — RQ3 全部响应 + 指标 + judge 判定
# - `results/human_spotcheck_sample.csv` — 80 条人工抽检样本
#
# **特性**：每条记录实时写入 CSV，支持断点续跑。
# Kernel 重启后再次运行，会自动跳过已完成的调用。

# %%
# ====== Cell 0-1: 安装依赖 ======
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "requests", "numpy", "pandas", "scipy", "datasets"])
print("依赖安装完成 ✓")

# %%
# ====== Cell 0-2: 设置工作目录 ======
import os
from pathlib import Path

# ★ 如果你的 notebook 不在 persona_hallucination/ 文件夹内，
#    取消下行注释并填写正确路径：
# os.chdir("/你的路径/persona_hallucination")

cwd = Path.cwd()
expected = ["config.py","api_client.py","prompts.py","data_prep.py","metrics.py","judge.py"]
missing = [f for f in expected if not (cwd/f).exists()]
if missing:
    print(f"⚠️  当前目录: {cwd}\n⚠️  缺少: {missing}")
else:
    print(f"当前目录: {cwd}\n所有文件就位 ✓")

# %%
# ====== Cell 0-3: 设置 API Key ======
os.environ["OPENROUTER_API_KEY"] = "0000000000000000000"  # ← 改成你的 key

from config import OPENROUTER_API_KEY
if OPENROUTER_API_KEY == "0000000000000000000":
    print("⚠️  请替换为真实 API Key！")
else:
    print(f"API Key 已设置 (前8位: {OPENROUTER_API_KEY[:8]}...) ✓")

# %%
# ====== Cell 0-4: 导入 & 初始化 ======
import json, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import config
from api_client import query_model
from prompts import PERSONA_CONDITIONS, ALL_CONDITION_IDS, USER_PROMPT_TEMPLATE
from judge import judge_factual_accuracy
from metrics import certainty_score, count_hedge_words

print(f"项目根目录: {config.PROJECT_ROOT}")
print(f"模型: {list(config.MODELS.keys())}")
print(f"条件数: {len(ALL_CONDITION_IDS)}")
print("导入成功 ✓")


# %% [markdown]
# ## 1. 数据准备

# %%
# ====== Cell 1: 下载 & 采样数据集 ======
from data_prep import prepare_triviaqa, prepare_popqa, prepare_medqa, save_dataset

if (config.DATA_DIR / "all_questions.json").exists():
    print("数据集已存在，跳过下载。")
    with open(config.DATA_DIR / "all_questions.json", encoding="utf-8") as f:
        all_questions = json.load(f)
else:
    tqa = prepare_triviaqa();  save_dataset(tqa, "triviaqa")
    popqa = prepare_popqa();   save_dataset(popqa, "popqa")
    medqa = prepare_medqa();   save_dataset(medqa, "medqa")
    all_questions = tqa + popqa + medqa
    save_dataset(all_questions, "all_questions")

print(f"总计: {len(all_questions)} 题")
for ds in ["triviaqa","popqa","medqa"]:
    print(f"  {ds}: {sum(1 for q in all_questions if q['dataset']==ds)}")


# %% [markdown]
# ## 2. 端到端冒烟测试（1 道题 × 1 个条件 × 1 个模型）

# %%
# ====== Cell 2: 冒烟测试 ======
test_q = all_questions[0]
print(f"题目: {test_q['question'][:80]}...")
try:
    r = query_model("gpt-4o-mini", "You are a general-purpose assistant.",
                     USER_PROMPT_TEMPLATE.format(question=test_q["question"]))
    print(f"✓ 模型调用成功 | 延迟 {r['latency_s']}s | {r['total_tokens']} tokens")
    print(f"  回复: {r['content'][:150]}...")

    j = judge_factual_accuracy(test_q["question"], test_q["gold_answer"], r["content"])
    print(f"✓ Judge 判定: {j['verdict']}")
    print(f"  CS={certainty_score(r['content']):.3f}, hedge={count_hedge_words(r['content'])}")
    print("\n冒烟测试通过 ✓")
except Exception as e:
    print(f"✗ 失败: {e}")


# %% [markdown]
# ## 3. RQ1 & RQ2: 静态 Persona 实验
#
# 全量: 400 题 × 9 条件 × 3 模型 × 10 重复 = **108,000** 次模型调用 + **108,000** 次 judge 调用
#
# **建议**: 先用 Cell 3-1 小规模验证，确认无误后运行 Cell 3-2 全量。
# 支持断点续跑：中断后重新执行即可自动跳过已完成部分。

# %%
# ====== Cell 3-1: 小规模验证 ======
import pandas as pd
from experiment_rq1_rq2 import run_rq12_pipeline, CSV_PATH as RQ12_CSV

# 先备份已有数据（如果有的话）
# 小规模测试也写入同一个 CSV，但因为有 resume 所以不会重复

run_rq12_pipeline(
    questions=all_questions[:3],             # 3 题
    condition_ids=["neutral_none", "authority_strong"],  # 2 条件
    model_keys=["gpt-4o-mini"],              # 1 模型
    repeats=1,                               # 1 重复
)

# 查看结果
if RQ12_CSV.exists():
    df = pd.read_csv(RQ12_CSV)
    print(f"\n当前已有 {len(df)} 条记录")
    print(df[["qid","condition_id","model_key","certainty_score",
              "hedge_count","word_count","judge_verdict"]].to_string(index=False))
print("\n小规模验证完成 ✓")

# %%
# ====== Cell 3-2: 全量 RQ1/RQ2 ======
# ⚠️ 约 216,000 次 API 调用，预计数小时
# ⚠️ 支持断点续跑：中断后重新执行会跳过已完成的部分
#
# 你也可以分模型运行，例如：
#   model_keys=["gpt-4o-mini"]   → 先跑一个模型
#   model_keys=["claude-3-haiku"] → 再跑下一个

run_rq12_pipeline(
    questions=all_questions,
    condition_ids=ALL_CONDITION_IDS,    # 全部 9 条件
    model_keys=list(config.MODELS.keys()),  # 全部 3 模型
    repeats=config.REPEAT_PER_CONDITION,    # 10 重复
)

# %%
# ====== Cell 3-3: 生成人工抽检样本 ======
from experiment_rq1_rq2 import generate_spotcheck_sample
generate_spotcheck_sample(n=80)
print("请在 results/human_spotcheck_sample.csv 中填写 human_verdict 列")


# %% [markdown]
# ## 4. RQ3: Persona 切换实验

# %%
# ====== Cell 4-1: 小规模 RQ3 验证 ======
from experiment_rq3 import run_rq3_pipeline, CSV_PATH as RQ3_CSV

run_rq3_pipeline(
    questions=all_questions,
    pair_indices=[0],            # 只测第 1 个 pair
    warmup_lengths=[5],          # 只测 k=5
    model_keys=["gpt-4o-mini"],
    n_post_switch_turns=3,       # 切换后只问 3 题
)

if RQ3_CSV.exists():
    df3 = pd.read_csv(RQ3_CSV)
    print(f"\n当前已有 {len(df3)} 条 RQ3 记录")
    print(df3.groupby("condition")["judge_verdict"].value_counts())
print("\nRQ3 小规模验证完成 ✓")

# %%
# ====== Cell 4-2: 全量 RQ3 ======
# 3 pair × 3 warmup × 3 model × 3 条件 × 20 题/条件 = 1,620+ 对话轮次
# 加上 judge 和 purity 调用，总计约 5,000-10,000 API 调用

run_rq3_pipeline(
    questions=all_questions,
    pair_indices=None,           # 全部 3 对
    warmup_lengths=None,         # [5, 10, 20]
    model_keys=None,             # 全部 3 模型
    n_post_switch_turns=20,
)


# %% [markdown]
# ## 5. 数据文件检查

# %%
# ====== Cell 5: 最终检查 ======
print("="*60)
print("  第一阶段完成 — 数据文件清单")
print("="*60)

for name, path in [
    ("RQ1/RQ2 响应数据", config.RESULTS_DIR / "rq1_rq2_responses.csv"),
    ("RQ3 响应数据",     config.RESULTS_DIR / "rq3_responses.csv"),
    ("人工抽检样本",     config.RESULTS_DIR / "human_spotcheck_sample.csv"),
]:
    if path.exists():
        size = path.stat().st_size / 1024
        try:
            n = len(pd.read_csv(path))
        except:
            n = "?"
        print(f"  ✓ {name}: {path.name} ({n} 行, {size:.0f} KB)")
    else:
        print(f"  ✗ {name}: 未生成")

print("\n下一步: 打开 stage2_analysis.py 进行统计分析。")
print("数据文件可以随时用 Excel/Python/R 打开做额外分析。")
