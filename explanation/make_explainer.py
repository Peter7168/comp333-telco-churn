"""Build explanation/project_explainer.pdf, a 5-minute intro to the project.

Run from the repo root after the pipeline:  python explanation/make_explainer.py
Numbers are read from reports/tables, so the PDF stays in sync with the results.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.config import FIGURES, TABLES  # noqa: E402

OUT = ROOT / "explanation" / "project_explainer.pdf"
MODEL_NAMES = {"hist_gb": "gradient boosting", "logreg": "logistic regression",
               "random_forest": "random forest", "knn": "k-nearest neighbours"}
DIAGRAM = ROOT / "explanation" / "pipeline_diagram.png"

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
LIGHT = colors.HexColor("#eef4fc")
LINE = colors.HexColor("#d9d8d4")

# I use DejaVu because the built-in PDF fonts have no arrow or bullet glyphs.
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))

H1 = ParagraphStyle("h1", fontName="Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Body", fontSize=10.5, leading=14, textColor=INK_2, spaceAfter=14)
H2 = ParagraphStyle("h2", fontName="Bold", fontSize=14, leading=18, textColor=BLUE,
                    spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.5, textColor=INK,
                      alignment=TA_LEFT, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11.5, textColor=INK_2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Bold")
BOX = ParagraphStyle("box", parent=BODY, fontSize=10.5, leading=15)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [p(f"•&nbsp;&nbsp;{t}", ParagraphStyle("b", parent=BODY, leftIndent=12,
                                                    firstLineIndent=-10, spaceAfter=3))
            for t in items]


def table(rows, widths, header=True):
    data = [[p(str(c), CELL_B if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 1, BLUE)]
    t.setStyle(TableStyle(style))
    return t


def callout(text):
    t = Table([[p(text, BOX)]], colWidths=[7 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("TOPPADDING", (0, 0), (-1, -1), 9),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return t


def figure(path, width):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width, width * ratio
    img.hAlign = "LEFT"
    return img


def draw_pipeline():
    """I draw the pipeline as boxes and arrows so a newcomer sees the whole flow at once."""
    steps = [
        ("8 raw files", "3 CSV\n+ 5 XLSX\n2 sources"),
        ("Bronze", "files loaded\nas they are"),
        ("Integrate", "one schema,\njoin, resolve\nconflicts"),
        ("Clean", "fill nulls,\ncap outliers"),
        ("Transform\n+ ETL", "scale, encode,\nnew features"),
        ("Gold", "model-ready\ntable"),
        ("Model", "predict\nchurn"),
        ("Dashboard", "charts +\nlive prediction"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.set_xlim(0, len(steps) * 1.4)
    ax.set_ylim(0, 2.3)
    ax.axis("off")
    for i, (title, desc) in enumerate(steps):
        x = i * 1.4 + 0.1
        lake = title in ("Bronze", "Gold")
        box = FancyBboxPatch((x, 0.35), 1.2, 1.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor="#2a78d6" if lake else "#eef4fc",
                             edgecolor="#2a78d6", linewidth=1.2)
        ax.add_patch(box)
        color = "white" if lake else "#0b0b0b"
        ax.text(x + 0.6, 1.55, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=color)
        ax.text(x + 0.6, 0.85, desc, ha="center", va="center", fontsize=8,
                color="white" if lake else "#52514e", linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.39, 1.12), xytext=(x + 1.21, 1.12),
                        arrowprops={"arrowstyle": "-|>", "color": "#52514e", "lw": 1.2})
    ax.text(0.1, 2.15, "Blue boxes are stored in Delta Lake (bronze = raw, silver = integrated "
            "and cleaned, gold = final features).", fontsize=8.5, color="#52514e")
    fig.savefig(DIAGRAM, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(INK_2)
    canvas.drawString(0.75 * inch, 0.5 * inch, "COMP 333 · Stats and Furious · Telco customer churn")
    canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    draw_pipeline()
    stages = pd.read_csv(TABLES / "exp_stages.csv")
    models = pd.read_csv(TABLES / "exp_models.csv").set_index("model")
    final = pd.read_csv(TABLES / "final_model_metrics.csv").iloc[0]
    stress = pd.read_csv(TABLES / "exp_stress.csv").set_index("setup")
    nulls = pd.read_csv(TABLES / "exp_nulls.csv").set_index("strategy")
    scalers = pd.read_csv(TABLES / "exp_scalers.csv").set_index("setup")
    stats = pd.read_csv(TABLES / "dataset_stats.csv")
    best = models["f1"].idxmax()
    W = 7 * inch

    s = []
    # ---------- page 1: what and why ----------
    s.append(p("Predicting which telecom customers will leave", H1))
    s.append(p("COMP 333 Data Analytics · Team Stats and Furious · A 5-minute introduction "
               "to the whole project", SUB))
    s.append(callout(
        "<b>In one sentence:</b> we combine customer data from two sources, clean and prepare "
        "it, and train a model that flags customers who are likely to cancel, so the company "
        "can offer them a deal before they go. A dashboard shows the results and scores new "
        "customers live."))

    s.append(p("The goal", H2))
    s.append(p("A phone and internet company loses about 1 in 4 customers (26.5%). Keeping a "
               "customer is much cheaper than finding a new one, so the company wants to know "
               "<b>who is about to leave</b>. That is our question."))
    s.append(p("For the course, the goal is also to show every step of a data pipeline: "
               "collecting data, integrating sources, cleaning, transforming, building features, "
               "storing in a data lake, modelling, and showing it all in a dashboard. For "
               "cleaning and transformation we also measure the model with and without each "
               "step."))

    s.append(p("The machine learning task", H2))
    s.append(table([
        ["Question", "Answer"],
        ["What do we predict?", "<b>Churn</b>: did the customer cancel? (Yes = 1, No = 0)"],
        ["What kind of task?", "<b>Binary classification</b>: the answer is a category, not a number"],
        ["Input", "One row per customer: contract, services, charges, age, location, ..."],
        ["How we score it", "<b>F1 score</b> on the churn class, plus ROC-AUC. We don't use plain "
                            "accuracy because only 26.5% churn: a model that always says \"stays\" "
                            "would already be 73.5% accurate and useless."],
    ], [1.6 * inch, 5.4 * inch]))

    s.append(p("The data", H2))
    s.append(p("Both sources describe the same 7,043 fictional customers of a California telecom, "
               "but they were published separately by IBM with different column names, formats "
               "and some disagreeing values. That is what makes integration a real problem."))
    s.append(table([
        ["Source", "Files", "Rows"],
        ["<b>1. IBM sample on GitHub</b> (CSV)",
         "customer_profiles, service_subscriptions, account_billing", "7,043 each"],
        ["<b>2. IBM Cognos 11.1.3 module</b> (XLSX)",
         "demographics, location, population, services, status",
         "7,043 each (population: 1,671 zip codes)"],
    ], [2.3 * inch, 2.9 * inch, 1.8 * inch]))
    s.append(p(f"{len(stats)} files, {stats['size_mb'].sum():.2f} MB in total. Source 1 is the "
               "file from Phase 1, split into 3 tables. Source 2 adds age, number of dependants, "
               "referrals, offers, internet type, location and zip-code population.", SMALL))

    # ---------- page 2: how it works ----------
    s.append(PageBreak())
    s.append(p("How it works", H2))
    s.append(p("One command, <font name='Bold'>python -m src.pipeline</font>, runs every step "
               "below in order. Each step is a small Python file in the <font name='Bold'>src/"
               "</font> folder."))
    s.append(figure(DIAGRAM, W))
    s.append(Spacer(1, 6))

    s.append(p("How we built it, step by step", H2))
    steps = [
        ("1. Collect + store", "ingest.py, lake.py",
         "Load all 8 files into the <b>bronze</b> layer of a <b>Delta Lake</b> (a data lake format "
         "that keeps every version of a table). A SQLite copy feeds the dashboard."),
        ("2. Integrate", "integrate.py, dependencies.py",
         "Rename every column to one shared name (74 source columns mapped), turn Yes/No into "
         "1/0 everywhere, and join on customer ID (1-to-1) and zip code (many-to-1). Every "
         "disagreement between sources is logged. We also check <b>functional dependencies</b>, "
         "e.g. zip code determines city, and age determines senior citizen."),
        ("3. Clean", "clean.py",
         "<b>Nulls:</b> 11 blank total charges (filled from source 2) and structural blanks such "
         "as \"no offer\" (filled with \"None\"). 6 strategies compared. <b>Outliers:</b> found with "
         "IQR, z-score and Isolation Forest; 5 strategies compared (keep, remove, cap, ...)."),
        ("4. Transform", "transform.py",
         "<b>Normalization</b> of skewed columns (log, Yeo-Johnson, quantile). <b>Scaling</b> (z-score, "
         "min-max, robust, max-abs, decimal scaling). <b>Encoding</b> of text categories (one-hot, "
         "ordinal, frequency, target)."),
        ("5. ETL features", "features.py, etl.py",
         "Build new columns such as number of add-ons, price per service, tenure band and "
         "auto-pay. Rank all features by how much they tell us about churn."),
        ("6. Experiments", "experiments.py",
         "Score every option with 5-fold cross-validation, changing <b>one setting at a time</b>, "
         "so we can say exactly what each step is worth."),
        ("7. Model + deploy", "train.py, predict.py, dashboard/app.py",
         "Train the best model, save it, and serve it in a Streamlit dashboard where you can "
         "enter a customer and get a churn probability."),
    ]
    s.append(table([["Step", "Code", "What happens"]] + [list(x) for x in steps],
                   [1.25 * inch, 1.45 * inch, 4.3 * inch]))

    s.append(KeepTogether([p("Three decisions worth knowing", H2)] + bullets([
        "<b>Source 1 wins conflicts.</b> When the sources disagree (e.g. contract type for 265 "
        "customers), we keep source 1, because the Phase 1 work and the churn label are built "
        "on it.",
        "<b>Some columns give away the answer, so we removed them.</b> Source 2 has a "
        "satisfaction score where 1-2 always means churn, plus churn reason and churn score. "
        "A model trained on them would look perfect but be useless in real life.",
        "<b>Rows are never deleted from the test data.</b> Cleaning is learned on training rows "
        "only, then applied to test rows, so the scores are honest.",
    ])))

    # ---------- outputs ----------
    s.append(p("What came out of it", H2))
    s.append(figure(FIGURES / "01_stages_f1.png", 5.2 * inch))
    s.append(p("The chart adds one project stage at a time. Most of the gain comes from "
               "<b>integrating the second source</b>.", SMALL))

    s.append(p("Key results", H2))
    drop = nulls.loc["drop_rows"]
    s.extend(bullets([
        f"<b>Final model:</b> {MODEL_NAMES.get(best, best)} with F1 "
        f"{models.loc[best, 'f1']:.3f} and ROC-AUC {models.loc[best, 'roc_auc']:.3f} in "
        f"cross-validation; F1 {final['f1']:.3f} on a separate 20% hold-out set.",
        f"<b>Integration:</b> F1 went from {stages['f1'].iloc[2]:.3f} to {stages['f1'].iloc[3]:.3f} "
        "after adding source 2.",
        "<b>Cleaning</b> changes little on the real data (only 11 bad cells). On a copy with "
        f"simulated damage, F1 fell from {stress.loc['clean data (reference)', 'f1']:.3f} to "
        f"{stress.loc['nulls: drop_rows', 'f1']:.3f}; imputing and capping outliers brought it "
        f"back to {stress.loc['outliers: cap_iqr', 'f1']:.3f}.",
        f"<b>Dropping rows with nulls</b> scores {drop['f1']:.3f} but only because it also "
        f"deletes {1 - drop['test_rows'] / nulls['test_rows'].max():.0%} of the customers we "
        "test on. It can't predict for them at all, so we don't use it.",
        f"<b>Scaling</b> barely matters for logistic regression but lifts KNN from "
        f"{scalers.loc['none | knn', 'f1']:.3f} to {scalers.loc['standard | knn', 'f1']:.3f}.",
        "<b>Strongest signals:</b> number of referrals, month-to-month contract, number of "
        "dependants and tenure.",
    ]))

    s.append(KeepTogether([p("Where to find things", H2), table([
        ["Folder / file", "What is inside"],
        ["data/raw/", "The 8 original files from both sources"],
        ["data/lake/", "Delta Lake tables: bronze, silver, gold (rebuilt on every run)"],
        ["src/", "All pipeline code, one file per step"],
        ["reports/tables/", "Every result as CSV: conflicts, dependencies, nulls, outliers, experiments"],
        ["reports/figures/", "The charts used in the slides"],
        ["dashboard/app.py", "The Streamlit dashboard"],
        ["docs/", "ERD and the presentation plan"],
    ], [1.8 * inch, 5.2 * inch])]))

    s.append(KeepTogether([p("Try it", H2), table([
        ["Command", "What it does"],
        ["pip install -r requirements.txt", "Installs pandas, scikit-learn, deltalake, streamlit, ..."],
        ["python -m src.pipeline", "Runs everything (about 8 minutes; add --skip-experiments for 1 minute)"],
        ["streamlit run dashboard/app.py", "Opens the dashboard in your browser"],
    ], [2.6 * inch, 4.4 * inch])]))

    s.append(KeepTogether([p("Words you will hear", H2), table([
        ["Term", "Meaning"],
        ["Churn", "A customer cancelling their service"],
        ["F1 score", "Balance of precision (flagged customers who really leave) and recall "
                     "(leavers we catch). 1.0 is perfect"],
        ["Cross-validation", "Split the data into 5 parts, train on 4, test on 1, repeat 5 times, "
                             "and average the scores"],
        ["IQR outlier", "A value far outside the middle 50% of the data (beyond 1.5 × IQR)"],
        ["One-hot encoding", "Turn a text column like Contract into one 0/1 column per value"],
        ["Delta Lake", "A data lake format that versions tables, so we can read any earlier copy"],
        ["Functional dependency", "X → Y: knowing X always tells you Y (zip code → city)"],
    ], [1.6 * inch, 5.4 * inch])]))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="COMP 333 project explainer", author="Stats and Furious")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
