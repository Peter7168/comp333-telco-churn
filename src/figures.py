"""Static charts for the slides (PNG). The dashboard has the interactive versions."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import lake
from src.config import FIGURES, TABLES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def hbar(ax, labels, values, color=BLUE, fmt="{:.3f}", xmax=None):
    y = range(len(labels))
    ax.barh(list(y), values, color=color, height=0.55)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    top = xmax or max(values) * 1.15
    ax.set_xlim(0, top)
    for i, v in zip(y, values):
        ax.text(v + top * 0.01, i, fmt.format(v), va="center", color=INK_2, fontsize=10)


def dots(ax, labels, values, highlight=None, std=None):
    """Dot plot for close scores, where bars from zero would hide the differences."""
    y = list(range(len(labels)))
    colors = [ORANGE if lab == highlight else BLUE for lab in labels]
    if std is not None:
        ax.errorbar(values, y, xerr=std, fmt="none", ecolor=GRID, elinewidth=2, zorder=1)
    ax.scatter(values, y, s=70, c=colors, zorder=2, edgecolors=SURFACE, linewidths=2)
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.4, -0.8)   # I leave room above the first label
    ax.grid(axis="y", visible=False)
    for i, v in zip(y, values):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=INK_2, fontsize=9)


def stages():
    t = pd.read_csv(TABLES / "exp_stages.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    hbar(ax, t["stage"], t["f1"], xmax=0.8)
    ax.set_title("F1 as each project stage is added")
    ax.set_xlabel("F1 on the churn class (logistic regression, 5-fold CV)")
    save(fig, "01_stages_f1")


def ablation(name, label_col, title, final_label):
    t = pd.read_csv(TABLES / f"exp_{name}.csv")
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(t) + 1.3))
    # I flag setups that were scored on fewer test rows (drop_rows), since that is not a fair comparison.
    full = t["test_rows"].max()
    labels = [f"{lab} ({r} test rows*)" if r < full else lab
              for lab, r in zip(t[label_col], t["test_rows"])]
    final_label = next(l for l in labels if l.startswith(final_label))
    dots(ax, labels, t["f1"], highlight=final_label, std=t["f1_std"])
    ax.set_title(title)
    ax.set_xlabel("F1 (dot = mean of 5 folds, gray bar = ±1 std). Orange = our choice.")
    save(fig, f"02_{name}")


def scalers():
    t = pd.read_csv(TABLES / "exp_scalers.csv")
    t["scaler_name"] = t["scaler"]
    lr, knn = t[t["model"] == "logreg"], t[t["model"] == "knn"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(lr))
    w = 0.38
    ax.bar([i - w / 2 for i in x], lr["f1"], width=w - 0.03, color=BLUE, label="Logistic regression")
    ax.bar([i + w / 2 for i in x], knn["f1"], width=w - 0.03, color=ORANGE, label="KNN (k=25)")
    for i, (a, b) in enumerate(zip(lr["f1"], knn["f1"])):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9, color=INK_2)
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(list(x), lr["scaler_name"])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("F1 score")
    ax.set_title("Scaling barely matters for logistic regression, a lot for KNN")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    save(fig, "03_scalers")


def stress():
    t = pd.read_csv(TABLES / "exp_stress.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{s}  ({r} test rows*)" if r < t["test_rows"].max() else s
              for s, r in zip(t["setup"], t["test_rows"])]
    dots(ax, labels, t["f1"], highlight="outliers: cap_iqr", std=t["f1_std"])
    ax.axvline(t["f1"].iloc[0], color=INK_2, linestyle="--", linewidth=1)
    ax.set_title("Stress test on simulated dirty data")
    ax.set_xlabel("F1. 10% blanks + 3% typos (x10) in 5 columns. Dashed = real clean data.")
    save(fig, "04_stress_test")


def churn_rates(gold):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, col, title in zip(axes, ["contract", "internet_type", "tenure_band"],
                              ["By contract", "By internet type (source 2)", "By tenure band (ETL)"]):
        rates = gold.groupby(col)["churn"].mean().sort_values(ascending=False) * 100
        hbar(ax, rates.index, rates.values, fmt="{:.0f}%", xmax=60)
        ax.set_title(title, fontsize=12)
    axes[0].set_xlabel("Churn rate (%)")
    fig.suptitle("Who churns? Share of customers who left, per group", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "05_churn_rates")


def boxplots(df):
    cols = ["tenure", "monthly_charges", "total_charges", "avg_monthly_gb_download",
            "total_long_distance_charges", "population"]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 3.4))
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), widths=0.5, patch_artist=True,
                   boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
                   medianprops={"color": BLUE, "linewidth": 2},
                   whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": ORANGE,
                               "markeredgecolor": ORANGE, "alpha": 0.5})
        ax.set_xticks([])
        ax.set_title(col.replace("_", " "), fontsize=10, fontweight="normal")
    fig.suptitle("IQR outliers (orange points). Source 1 columns have none; source 2 columns do",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "06_outlier_boxplots")


def importance():
    t = pd.read_csv(TABLES / "final_model_importance.csv").head(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hbar(ax, t["feature"], t["f1_drop"])
    ax.set_title("What the final model relies on (permutation importance)")
    ax.set_xlabel("Drop in F1 when the column is shuffled")
    save(fig, "07_feature_importance")


def run():
    gold = lake.read("gold", "customer_features")
    integrated = lake.read("silver", "customers_integrated")
    stages()
    ablation("nulls", "strategy", "Null handling strategies", "cross_source")
    ablation("outliers", "strategy", "Outlier handling strategies", "cap_iqr")
    ablation("normalizers", "normalizer", "Normalizing the skewed columns", "yeo_johnson")
    ablation("encoders", "encoder", "Encoding the categorical columns", "onehot")
    ablation("models", "model", "Models on the final prepared data", "hist_gb")
    scalers()
    stress()
    churn_rates(gold)
    boxplots(integrated)
    importance()
    print(f"saved figures to {FIGURES}")


if __name__ == "__main__":
    run()
