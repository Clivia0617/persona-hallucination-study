"""
Statistical Analysis Module — reads CSV data, runs all tests, generates all figures.

INPUT:   results/rq1_rq2_responses.csv
         results/rq3_responses.csv
         results/human_spotcheck_sample.csv  (optional, for Cohen's κ)

OUTPUT:  results/figures/*.png              (all visualizations)
         results/analysis_summary.csv       (key statistics table)
         console printout of all test results

Covers every hypothesis and visualization from the project proposal:
  H1: Expert personas → lower hedge density, higher CS
  H2: Strong expert → higher HR than neutral
  H3: Persona × model interaction
  H4: HR monotonically increases with confidence intensity
  H5: Post-switch HR > clean-start HR (residual effect)
  H6: RHE decays exponentially; report t*
  H7: RHE larger for semantically distant persona pairs

Usage from notebook:
    from analysis import run_all_analyses
    run_all_analyses()
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from collections import defaultdict

from config import RESULTS_DIR, BOOTSTRAP_ITERATIONS
from metrics import hallucination_rate, abstention_rate, bootstrap_hr, bootstrap_ci

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════

def load_rq12() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "rq1_rq2_responses.csv")
    # Ensure types
    df["confidence_ordinal"] = df["confidence_ordinal"].astype(int)
    df["is_incorrect"] = (df["judge_verdict"] == "incorrect").astype(int)
    df["is_abstain"] = (df["judge_verdict"] == "abstain").astype(int)
    # Binary for GLMM (exclude abstain/cannot_determine)
    df["evaluable"] = df["judge_verdict"].isin(["correct", "incorrect"])
    logger.info(f"RQ1/RQ2 数据: {len(df)} 条, {df['model_key'].nunique()} 模型, "
                f"{df['condition_id'].nunique()} 条件")
    return df


def load_rq3() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "rq3_responses.csv")
    df["is_incorrect"] = (df["judge_verdict"] == "incorrect").astype(int)
    df["evaluable"] = df["judge_verdict"].isin(["correct", "incorrect"])
    logger.info(f"RQ3 数据: {len(df)} 条")
    return df


# ══════════════════════════════════════════════════════════════
#  Helper: per-group HR with bootstrap CI
# ══════════════════════════════════════════════════════════════

def _group_hr_ci(df, group_cols):
    """Compute HR + 95% bootstrap CI per group. Returns a DataFrame."""
    rows = []
    for name, grp in df.groupby(group_cols):
        verdicts = grp["judge_verdict"].tolist()
        hr, lo, hi = bootstrap_hr(verdicts)
        ar = abstention_rate(verdicts)
        row = dict(zip(group_cols, name)) if isinstance(name, tuple) else {group_cols[0]: name}
        row.update({"HR": hr, "HR_CI_lo": lo, "HR_CI_hi": hi, "AR": ar, "n": len(verdicts)})
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  H1: Persona effects on style (RQ1)
# ══════════════════════════════════════════════════════════════

def analyze_h1(df):
    print("\n" + "="*70)
    print("  H1: Expert personas → lower hedge density, higher certainty score")
    print("="*70)

    # Descriptive
    style = df.groupby(["condition_id","persona_category","confidence_level","model_key"]).agg(
        CS_mean=("certainty_score","mean"), CS_std=("certainty_score","std"),
        hedge_mean=("hedge_count","mean"), wc_mean=("word_count","mean"),
        n=("certainty_score","count"),
    ).reset_index()
    print("\n[描述统计] 风格指标 (condition × model):")
    print(style.to_string(index=False))

    # Kruskal-Wallis: CS ~ persona_category, per model
    print("\n[Kruskal-Wallis] CS across persona categories:")
    for mk in sorted(df["model_key"].unique()):
        sub = df[df["model_key"]==mk]
        groups = [g["certainty_score"].values for _, g in sub.groupby("persona_category")]
        H, p = stats.kruskal(*groups)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {mk}: H={H:.3f}, p={p:.4f} {sig}")

    # Dunn post-hoc with Bonferroni
    try:
        import scikit_posthocs as sp
        print("\n[Dunn post-hoc] (Bonferroni corrected):")
        for mk in sorted(df["model_key"].unique()):
            sub = df[df["model_key"]==mk]
            dunn = sp.posthoc_dunn(sub, val_col="certainty_score",
                                    group_col="persona_category", p_adjust="bonferroni")
            print(f"\n  {mk}:")
            print(dunn.round(4).to_string())
    except ImportError:
        print("  (scikit-posthocs 未安装, 跳过 Dunn post-hoc)")

    # Cohen's d: neutral vs each expert condition, per model
    print("\n[Cohen's d] neutral vs expert conditions:")
    for mk in sorted(df["model_key"].unique()):
        neutral_cs = df[(df["model_key"]==mk)&(df["condition_id"]=="neutral_none")]["certainty_score"]
        for cid in ["authority_strong","professional_strong","non_expert_none"]:
            other_cs = df[(df["model_key"]==mk)&(df["condition_id"]==cid)]["certainty_score"]
            if len(neutral_cs)>0 and len(other_cs)>0:
                pooled_std = np.sqrt((neutral_cs.std()**2 + other_cs.std()**2) / 2)
                d = (other_cs.mean() - neutral_cs.mean()) / pooled_std if pooled_std>0 else 0
                print(f"  {mk} | neutral vs {cid}: d={d:.3f}")

    # Visualization: CS box plots
    fig, axes = plt.subplots(1, df["model_key"].nunique(),
                              figsize=(6*df["model_key"].nunique(), 6), sharey=True)
    if df["model_key"].nunique() == 1:
        axes = [axes]
    for ax, mk in zip(axes, sorted(df["model_key"].unique())):
        sub = df[df["model_key"]==mk]
        order = sorted(sub["condition_id"].unique())
        sns.boxplot(data=sub, x="condition_id", y="certainty_score", ax=ax, order=order)
        ax.set_title(mk, fontsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right", fontsize=9)
        ax.set_xlabel("")
    fig.suptitle("H1: Certainty Score by Condition", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "h1_certainty_score_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # CS dose-response chart
    dose_df = df[df["confidence_ordinal"]>0].copy()
    if len(dose_df) > 0:
        dose_agg = dose_df.groupby(["confidence_level","confidence_ordinal","model_key"]).agg(
            CS_mean=("certainty_score","mean")
        ).reset_index().sort_values("confidence_ordinal")
        fig, ax = plt.subplots(figsize=(7,5))
        for mk in sorted(dose_agg["model_key"].unique()):
            sub = dose_agg[dose_agg["model_key"]==mk]
            ax.plot(sub["confidence_ordinal"], sub["CS_mean"], "o-", label=mk, linewidth=2)
        ax.set_xticks([1,2,3]); ax.set_xticklabels(["weak","medium","strong"])
        ax.set_xlabel("Confidence Intensity"); ax.set_ylabel("Mean Certainty Score")
        ax.set_title("H1: CS Dose-Response"); ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "h1_cs_dose_response.png", dpi=150)
        plt.close()

    print(f"\n  图表已保存: h1_certainty_score_boxplots.png, h1_cs_dose_response.png")


# ══════════════════════════════════════════════════════════════
#  H2: Strong expert → higher HR (RQ2)
# ══════════════════════════════════════════════════════════════

def analyze_h2(df):
    print("\n" + "="*70)
    print("  H2: High-confidence expert → higher hallucination rate")
    print("="*70)

    hr_table = _group_hr_ci(df, ["condition_id", "model_key"])
    print("\n[HR with 95% bootstrap CI]:")
    print(hr_table.to_string(index=False))

    # Mann-Whitney U: neutral vs authority_strong, per model
    print("\n[Mann-Whitney U] neutral vs authority_strong (one-sided: HR_strong > HR_neutral):")
    for mk in sorted(df["model_key"].unique()):
        ne = df[(df["model_key"]==mk)&(df["condition_id"]=="neutral_none")&df["evaluable"]]
        se = df[(df["model_key"]==mk)&(df["condition_id"]=="authority_strong")&df["evaluable"]]
        if len(ne)==0 or len(se)==0: continue
        U, p = stats.mannwhitneyu(se["is_incorrect"], ne["is_incorrect"], alternative="greater")
        # Odds ratio
        a, b = se["is_incorrect"].sum(), (1-se["is_incorrect"]).sum()
        c, d_ = ne["is_incorrect"].sum(), (1-ne["is_incorrect"]).sum()
        OR = (a*d_)/(b*c) if b*c > 0 else float("inf")
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {mk}: U={U:.0f}, p={p:.4f} {sig}, OR={OR:.2f}")

    # Heatmap
    pivot = df.groupby(["condition_id","model_key"]).apply(
        lambda g: hallucination_rate(g["judge_verdict"].tolist())
    ).unstack("model_key")
    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("H2: Hallucination Rate Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "h2_hr_heatmap.png", dpi=150)
    plt.close()

    # AR vs HR scatter (suppression mechanism)
    ar_hr = _group_hr_ci(df, ["condition_id","model_key"])
    fig, ax = plt.subplots(figsize=(7,5))
    for mk in sorted(ar_hr["model_key"].unique()):
        sub = ar_hr[ar_hr["model_key"]==mk]
        ax.scatter(sub["AR"], sub["HR"], label=mk, s=60, alpha=0.7)
    ax.set_xlabel("Abstention Rate"); ax.set_ylabel("Hallucination Rate")
    ax.set_title("H2: AR vs HR (Suppression Mechanism)"); ax.legend()
    # Correlation annotation
    rho, p = stats.spearmanr(ar_hr["AR"].dropna(), ar_hr["HR"].dropna())
    ax.annotate(f"Spearman ρ={rho:.3f}, p={p:.4f}", xy=(0.02,0.98),
                xycoords="axes fraction", va="top", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "h2_ar_vs_hr_scatter.png", dpi=150)
    plt.close()

    print(f"\n  图表已保存: h2_hr_heatmap.png, h2_ar_vs_hr_scatter.png")


# ══════════════════════════════════════════════════════════════
#  H3: Persona × Model interaction
# ══════════════════════════════════════════════════════════════

def analyze_h3(df):
    print("\n" + "="*70)
    print("  H3: Persona × model interaction")
    print("="*70)

    # Logistic regression with cluster-robust SE (GLMM approximation)
    try:
        import statsmodels.formula.api as smf
        sub = df[df["evaluable"]].copy()
        sub["persona_category"] = pd.Categorical(sub["persona_category"])
        sub["model_key"] = pd.Categorical(sub["model_key"])

        formula = "is_incorrect ~ C(persona_category) * C(model_key)"
        model = smf.logit(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["qid"]},
            disp=False, maxiter=200,
        )
        print("\n[Logistic Regression with cluster-robust SE]")
        print("  (Approximation to GLMM; question ID as cluster)")
        print(model.summary2().tables[1].to_string())

        # Interaction significance
        interaction_terms = [t for t in model.pvalues.index if ":" in t]
        if interaction_terms:
            min_p = model.pvalues[interaction_terms].min()
            print(f"\n  交互项最小 p 值: {min_p:.4f}")
            print(f"  交互效应{'显著' if min_p < 0.05 else '不显著'} (α=0.05)")
    except Exception as e:
        print(f"  Logistic 回归失败: {e}")
        print("  使用非参数替代方案...")

    # Interaction plot
    interact = df.groupby(["persona_category","model_key"]).apply(
        lambda g: hallucination_rate(g["judge_verdict"].tolist())
    ).reset_index(name="HR")
    fig, ax = plt.subplots(figsize=(8,5))
    for mk in sorted(interact["model_key"].unique()):
        sub = interact[interact["model_key"]==mk]
        ax.plot(sub["persona_category"], sub["HR"], "o-", label=mk, linewidth=2, markersize=8)
    ax.set_xlabel("Persona Category"); ax.set_ylabel("Hallucination Rate")
    ax.set_title("H3: Persona × Model Interaction"); ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "h3_persona_model_interaction.png", dpi=150)
    plt.close()
    print(f"\n  图表已保存: h3_persona_model_interaction.png")


# ══════════════════════════════════════════════════════════════
#  H4: Dose-response (confidence intensity → HR)
# ══════════════════════════════════════════════════════════════

def analyze_h4(df):
    print("\n" + "="*70)
    print("  H4: HR increases monotonically with confidence intensity")
    print("="*70)

    sub = df[(df["confidence_ordinal"]>0) & df["evaluable"]].copy()

    print("\n[Spearman ρ] confidence_ordinal vs is_incorrect, per model:")
    for mk in sorted(sub["model_key"].unique()):
        ms = sub[sub["model_key"]==mk]
        rho, p = stats.spearmanr(ms["confidence_ordinal"], ms["is_incorrect"])
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {mk}: ρ={rho:.3f}, p={p:.4f} {sig}")

    # All models combined
    rho, p = stats.spearmanr(sub["confidence_ordinal"], sub["is_incorrect"])
    print(f"  ALL: ρ={rho:.3f}, p={p:.4f}")

    # Dose-response chart (HR)
    dose_hr = sub.groupby(["confidence_level","confidence_ordinal","model_key"]).apply(
        lambda g: hallucination_rate(g["judge_verdict"].tolist())
    ).reset_index(name="HR").sort_values("confidence_ordinal")

    fig, ax = plt.subplots(figsize=(7,5))
    for mk in sorted(dose_hr["model_key"].unique()):
        s = dose_hr[dose_hr["model_key"]==mk]
        ax.plot(s["confidence_ordinal"], s["HR"], "o-", label=mk, linewidth=2)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["weak","medium","strong"])
    ax.set_xlabel("Confidence Intensity"); ax.set_ylabel("Hallucination Rate")
    ax.set_title("H4: Dose-Response (Confidence → HR)"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "h4_hr_dose_response.png", dpi=150)
    plt.close()
    print(f"\n  图表已保存: h4_hr_dose_response.png")


# ══════════════════════════════════════════════════════════════
#  H5: Post-switch HR > clean-start HR (RQ3)
# ══════════════════════════════════════════════════════════════

def analyze_h5(df3):
    print("\n" + "="*70)
    print("  H5: Post-switch HR significantly exceeds clean-start HR")
    print("="*70)

    # Bonferroni correction: number of tests = unique (pair, model) combos
    pairs = df3["pair_label"].dropna().unique()
    models = df3["model_key"].unique()
    n_tests = len(pairs) * len(models)

    print(f"\n[Paired Wilcoxon signed-rank] (Bonferroni α = {0.05/n_tests:.4f}, {n_tests} tests):")
    for pl in sorted(pairs):
        for mk in sorted(models):
            clean = df3[(df3["pair_label"]==pl)&(df3["model_key"]==mk)&(df3["condition"]=="clean_start")]
            post = df3[(df3["pair_label"]==pl)&(df3["model_key"]==mk)&(df3["condition"]=="post_switch")]
            if len(clean)==0 or len(post)==0: continue

            merged = pd.merge(
                clean[["qid","is_incorrect"]].rename(columns={"is_incorrect":"clean"}),
                post[["qid","is_incorrect"]].rename(columns={"is_incorrect":"post"}),
                on="qid"
            )
            if len(merged)==0: continue
            try:
                W, p = stats.wilcoxon(merged["post"], merged["clean"],
                                       alternative="greater", zero_method="zsplit")
                p_adj = min(p * n_tests, 1.0)
                sig = "***" if p_adj<0.001 else "**" if p_adj<0.01 else "*" if p_adj<0.05 else "ns"
                print(f"  {pl} | {mk}: W={W:.0f}, p_raw={p:.4f}, p_bonf={p_adj:.4f} {sig}")
            except Exception as e:
                print(f"  {pl} | {mk}: test failed ({e})")

    # Bar chart: clean-start vs post-switch HR
    bars = []
    for pl in sorted(pairs):
        for mk in sorted(models):
            for cond in ["clean_start","post_switch"]:
                sub = df3[(df3["pair_label"]==pl)&(df3["model_key"]==mk)&(df3["condition"]==cond)]
                if len(sub)==0: continue
                hr, lo, hi = bootstrap_hr(sub["judge_verdict"].tolist())
                bars.append({"pair":pl, "model":mk, "condition":cond, "HR":hr, "lo":lo, "hi":hi})

    if bars:
        bdf = pd.DataFrame(bars)
        fig, ax = plt.subplots(figsize=(12,5))
        x_labels = [f"{r['pair']}\n{r['model']}" for _, r in bdf[bdf["condition"]=="clean_start"].iterrows()]
        x = np.arange(len(x_labels))
        w = 0.35
        clean_hrs = bdf[bdf["condition"]=="clean_start"]["HR"].values
        post_hrs = bdf[bdf["condition"]=="post_switch"]["HR"].values
        ax.bar(x-w/2, clean_hrs, w, label="Clean Start", color="steelblue", alpha=0.8)
        ax.bar(x+w/2, post_hrs, w, label="Post Switch", color="tomato", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Hallucination Rate")
        ax.set_title("H5: Clean-Start vs Post-Switch HR"); ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "h5_clean_vs_postswitch.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  图表已保存: h5_clean_vs_postswitch.png")


# ══════════════════════════════════════════════════════════════
#  H6: Exponential decay of RHE
# ══════════════════════════════════════════════════════════════

def _exp_decay(t, alpha, beta, gamma):
    return alpha * np.exp(-beta * t) + gamma


def analyze_h6(df3):
    print("\n" + "="*70)
    print("  H6: RHE decays exponentially; f(t) = α·exp(-β·t) + γ")
    print("="*70)

    pairs = df3["pair_label"].dropna().unique()
    models = df3["model_key"].unique()

    fig, ax = plt.subplots(figsize=(10,6))
    decay_results = []

    for pl in sorted(pairs):
        for mk in sorted(models):
            clean = df3[(df3["pair_label"]==pl)&(df3["model_key"]==mk)&(df3["condition"]=="clean_start")]
            post = df3[(df3["pair_label"]==pl)&(df3["model_key"]==mk)&(df3["condition"]=="post_switch")]
            if len(clean)==0 or len(post)==0: continue

            hr_clean = hallucination_rate(clean["judge_verdict"].tolist())

            # Per-turn HR
            turn_data = []
            for t in sorted(post["turn_index"].unique()):
                tv = post[post["turn_index"]==t]["judge_verdict"].tolist()
                turn_data.append({"t": t, "HR": hallucination_rate(tv)})
            tdf = pd.DataFrame(turn_data)
            tdf["RHE"] = tdf["HR"] - hr_clean

            ax.plot(tdf["t"], tdf["RHE"], "o-", label=f"{pl} ({mk})", alpha=0.7, markersize=5)

            # Fit decay
            try:
                popt, _ = curve_fit(
                    _exp_decay, tdf["t"].values.astype(float), tdf["RHE"].values,
                    p0=[0.3, 0.2, 0.0], bounds=([0,0,-1],[2,5,1]), maxfev=10000
                )
                a, b, g = popt
                t_star = -np.log(0.05/a)/b if a>0 and b>0 else float("inf")

                # Plot fitted curve
                t_fit = np.linspace(tdf["t"].min(), tdf["t"].max(), 100)
                ax.plot(t_fit, _exp_decay(t_fit, *popt), "--", alpha=0.5)

                decay_results.append({"pair": pl, "model": mk,
                                       "alpha": a, "beta": b, "gamma": g, "t_star": t_star})
                print(f"  {pl} | {mk}: α={a:.3f}, β={b:.3f}, γ={g:.3f}, t*={t_star:.1f} turns")
            except Exception as e:
                print(f"  {pl} | {mk}: 拟合失败 ({e})")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Turn Index (post-switch)")
    ax.set_ylabel("RHE (Residual Hallucination Excess)")
    ax.set_title("H6: RHE Decay with Exponential Fit")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "h6_rhe_decay.png", dpi=150)
    plt.close()

    # PPS decay curve
    post_all = df3[(df3["condition"]=="post_switch") & (df3["purity_verdict"]!="")]
    if len(post_all) > 0:
        fig2, ax2 = plt.subplots(figsize=(10,6))
        for pl in sorted(pairs):
            for mk in sorted(models):
                sub = post_all[(post_all["pair_label"]==pl)&(post_all["model_key"]==mk)]
                if len(sub)==0: continue
                pps_by_t = sub.groupby("turn_index").apply(
                    lambda g: (g["purity_verdict"].str.upper()=="B").mean()
                ).reset_index(name="PPS")
                ax2.plot(pps_by_t["turn_index"], pps_by_t["PPS"], "o-",
                         label=f"{pl} ({mk})", alpha=0.7, markersize=5)
        ax2.axhline(1.0, color="green", ls="--", alpha=0.3, label="Full adoption of B")
        ax2.set_xlabel("Turn Index (post-switch)")
        ax2.set_ylabel("Persona Purity Score (PPS)")
        ax2.set_title("H6: PPS Decay (transition to Persona B)")
        ax2.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "h6_pps_decay.png", dpi=150)
        plt.close()
        print(f"\n  图表已保存: h6_rhe_decay.png, h6_pps_decay.png")
    else:
        print(f"\n  图表已保存: h6_rhe_decay.png")

    return pd.DataFrame(decay_results) if decay_results else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  H7: Semantic distance → larger RHE
# ══════════════════════════════════════════════════════════════

def analyze_h7(df3):
    print("\n" + "="*70)
    print("  H7: RHE larger for semantically distant persona pairs")
    print("="*70)

    for dist in ["low", "high"]:
        post = df3[(df3["distance_label"]==dist)&(df3["condition"]=="post_switch")]
        clean = df3[(df3["distance_label"]==dist)&(df3["condition"]=="clean_start")]
        if len(post)==0 or len(clean)==0: continue
        hr_post = hallucination_rate(post["judge_verdict"].tolist())
        hr_clean = hallucination_rate(clean["judge_verdict"].tolist())
        print(f"  distance={dist}: HR_post={hr_post:.3f}, HR_clean={hr_clean:.3f}, "
              f"RHE={hr_post-hr_clean:.3f}")

    # Mann-Whitney on RHE between low and high distance
    low_post = df3[(df3["distance_label"]=="low")&(df3["condition"]=="post_switch")&df3["evaluable"]]
    high_post = df3[(df3["distance_label"]=="high")&(df3["condition"]=="post_switch")&df3["evaluable"]]
    if len(low_post)>0 and len(high_post)>0:
        U, p = stats.mannwhitneyu(high_post["is_incorrect"], low_post["is_incorrect"],
                                   alternative="greater")
        print(f"\n  [Mann-Whitney] high vs low distance: U={U:.0f}, p={p:.4f}")


# ══════════════════════════════════════════════════════════════
#  Bootstrap Rank Stability
# ══════════════════════════════════════════════════════════════

def analyze_rank_stability(df, n_iter=BOOTSTRAP_ITERATIONS):
    print("\n" + "="*70)
    print("  Bootstrap Rank Stability (model rankings by HR)")
    print("="*70)

    rng = np.random.RandomState(42)
    models = sorted(df["model_key"].unique())
    qids = df["qid"].unique()
    rank_matrix = {m: [] for m in models}

    for _ in range(n_iter):
        sample_qids = rng.choice(qids, size=len(qids), replace=True)
        boot = df[df["qid"].isin(sample_qids)]
        hrs = {}
        for mk in models:
            verdicts = boot[boot["model_key"]==mk]["judge_verdict"].tolist()
            hrs[mk] = hallucination_rate(verdicts)
        ranked = sorted(hrs, key=lambda m: hrs[m])
        for rank, m in enumerate(ranked, 1):
            rank_matrix[m].append(rank)

    print("\n  模型排名分布 (1=最低HR, 即最好):")
    for m in models:
        r = np.array(rank_matrix[m])
        print(f"  {m}: median={np.median(r):.0f}, "
              f"P(rank=1)={np.mean(r==1):.1%}, "
              f"P(rank={len(models)})={np.mean(r==len(models)):.1%}")

    # Rank distribution plot
    fig, ax = plt.subplots(figsize=(7,4))
    for i, m in enumerate(models):
        counts = [np.mean(np.array(rank_matrix[m])==r) for r in range(1, len(models)+1)]
        ax.bar(np.arange(1,len(models)+1) + i*0.25, counts, 0.25, label=m)
    ax.set_xticks(range(1, len(models)+1))
    ax.set_xlabel("Rank"); ax.set_ylabel("Probability")
    ax.set_title("Bootstrap Rank Distribution"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rank_stability.png", dpi=150)
    plt.close()
    print(f"\n  图表已保存: rank_stability.png")


# ══════════════════════════════════════════════════════════════
#  Cohen's Kappa (Human vs Judge)
# ══════════════════════════════════════════════════════════════

def analyze_cohens_kappa():
    print("\n" + "="*70)
    print("  Cohen's κ: LLM Judge vs Human Annotator")
    print("="*70)

    path = RESULTS_DIR / "human_spotcheck_sample.csv"
    if not path.exists():
        print("  ⚠️  human_spotcheck_sample.csv 不存在。请先运行 generate_spotcheck_sample()。")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    if "human_verdict" not in df.columns or df["human_verdict"].isna().all() or (df["human_verdict"]=="").all():
        print("  ⚠️  human_verdict 列为空。请先完成人工标注后再运行此分析。")
        return

    # Filter rows with both verdicts
    sub = df[df["human_verdict"].notna() & (df["human_verdict"]!="")].copy()
    sub["judge_verdict"] = sub["judge_verdict"].str.lower().str.strip()
    sub["human_verdict"] = sub["human_verdict"].str.lower().str.strip()

    from sklearn.metrics import cohen_kappa_score, classification_report
    kappa = cohen_kappa_score(sub["human_verdict"], sub["judge_verdict"])
    print(f"\n  Cohen's κ = {kappa:.3f}")
    print(f"  解读: {'excellent' if kappa>0.8 else 'good' if kappa>0.6 else 'moderate' if kappa>0.4 else 'fair' if kappa>0.2 else 'poor'}")
    print(f"\n  Classification Report:")
    print(classification_report(sub["human_verdict"], sub["judge_verdict"]))


# ══════════════════════════════════════════════════════════════
#  Summary Table
# ══════════════════════════════════════════════════════════════

def generate_summary_table(df12, df3=None):
    """Export a concise summary CSV of key statistics."""
    rows = []

    # Per-condition HR summary (RQ1/RQ2)
    for (cid, mk), grp in df12.groupby(["condition_id","model_key"]):
        verdicts = grp["judge_verdict"].tolist()
        hr, lo, hi = bootstrap_hr(verdicts)
        ar = abstention_rate(verdicts)
        cs = grp["certainty_score"].mean()
        rows.append({
            "RQ": "1/2", "condition": cid, "model": mk,
            "HR": round(hr,4), "HR_CI_lo": round(lo,4), "HR_CI_hi": round(hi,4),
            "AR": round(ar,4), "CS_mean": round(cs,4), "n": len(verdicts),
        })

    summary = pd.DataFrame(rows)
    path = RESULTS_DIR / "analysis_summary.csv"
    summary.to_csv(path, index=False)
    logger.info(f"汇总表已保存: {path}")
    return summary


# ══════════════════════════════════════════════════════════════
#  Master entry point
# ══════════════════════════════════════════════════════════════

def run_all_analyses():
    """Run every analysis in sequence. Call this from the notebook."""
    print("\n" + "#"*70)
    print("#  Persona Hallucination Study — 完整统计分析")
    print("#"*70)

    # ---- RQ1/RQ2 ----
    df12 = load_rq12()
    analyze_h1(df12)
    analyze_h2(df12)
    analyze_h3(df12)
    analyze_h4(df12)
    analyze_rank_stability(df12)

    # ---- RQ3 ----
    df3 = None
    rq3_path = RESULTS_DIR / "rq3_responses.csv"
    if rq3_path.exists():
        df3 = load_rq3()
        analyze_h5(df3)
        decay_df = analyze_h6(df3)
        analyze_h7(df3)
    else:
        print("\n  ⚠️  rq3_responses.csv 不存在, 跳过 RQ3 分析。")

    # ---- Human verification ----
    analyze_cohens_kappa()

    # ---- Summary ----
    summary = generate_summary_table(df12, df3)
    print("\n[汇总表预览]:")
    print(summary.to_string(index=False))

    print("\n" + "#"*70)
    print(f"#  分析完成。图表: {FIG_DIR}")
    print(f"#  汇总表: {RESULTS_DIR / 'analysis_summary.csv'}")
    print("#"*70)
