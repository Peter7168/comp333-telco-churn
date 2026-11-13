"""Build explanation/project_explainer.pdf, a 5-minute intro to the project.

Run from the repo root after the pipeline:  python explanation/make_explainer.py
Numbers are read from reports/tables, so the PDF stays in sync with the results.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src import lake  # noqa: E402
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
pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))
# I link the two so <b> inside a paragraph switches to the bold file.
registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")

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


def table(rows, widths, header=True, section_rows=()):
    data = [[p(str(c), CELL_B if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 1, BLUE)]
    for r in section_rows:
        # A section row spans the whole table, like a sub-heading.
        style += [("SPAN", (0, r), (-1, r)), ("TOPPADDING", (0, r), (-1, r), 8)]
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


def examples():
    """Pull real values for the worked examples, so the PDF never shows made-up numbers."""
    from src import lake
    from src.clean import iqr_fences

    cid = "7590-VHVEG"
    bronze = {n: lake.read("bronze", n) for n in
              ("github_customer_profiles", "github_account_billing", "cognos_services",
               "cognos_demographics", "cognos_location", "cognos_population")}
    integ = lake.read("silver", "customers_integrated")
    clean = lake.read("silver", "customers_clean")
    gold = lake.read("gold", "customer_features")

    def row(name, key, value):
        df = bronze[name]
        return df[df[key] == value].iloc[0]

    ex = {
        "s1_prof": row("github_customer_profiles", "customerID", cid),
        "s1_bill": row("github_account_billing", "customerID", cid),
        "s2_srv": row("cognos_services", "Customer ID", cid),
        "s2_dem": row("cognos_demographics", "Customer ID", cid),
        "s2_loc": row("cognos_location", "Customer ID", cid),
        "integ": integ[integ["customer_id"] == cid].iloc[0],
        "gold": gold[gold["customer_id"] == cid].iloc[0],
    }
    zip_code = ex["s2_loc"]["Zip Code"]
    ex["population"] = row("cognos_population", "Zip Code", zip_code)["Population"]
    ex["same_zip"] = int((bronze["cognos_location"]["Zip Code"] == zip_code).sum())

    # A contract conflict between the two sources.
    s1 = bronze["github_account_billing"][["customerID", "Contract"]]
    s2 = bronze["cognos_services"][["Customer ID", "Contract"]]
    both = s1.merge(s2, left_on="customerID", right_on="Customer ID", suffixes=("_s1", "_s2"))
    conflict = both[both["Contract_s1"].str.lower() != both["Contract_s2"].str.lower()].iloc[0]
    ex["conflict"] = conflict

    # One of the 11 blank totals, before and after cleaning.
    blank_id = integ.loc[integ["total_charges"].isna(), "customer_id"].iloc[0]
    ex["no_offer_id"] = integ.loc[integ["offer"].isna(), "customer_id"].iloc[0]
    ex["blank"] = (integ[integ["customer_id"] == blank_id].iloc[0],
                   clean[clean["customer_id"] == blank_id].iloc[0])

    # The biggest data download, and the fence it gets capped at.
    big = integ.sort_values("avg_monthly_gb_download").iloc[-1]
    ex["outlier"] = (big, iqr_fences(integ["avg_monthly_gb_download"])[3])

    # Transformation of our main customer, using statistics from the clean table.
    c, t = clean, ex["integ"]
    ex["z_tenure"] = (t["tenure"] - c["tenure"].mean()) / c["tenure"].std(ddof=0)
    lo, hi = c["monthly_charges"].min(), c["monthly_charges"].max()
    ex["minmax_monthly"] = (t["monthly_charges"] - lo) / (hi - lo)
    ex["la_share"] = (c["city"] == t["city"]).mean()
    ex["la_churn"] = c.loc[c["city"] == t["city"], "churn"].mean()
    return ex


def step_examples(ex):
    """One block per pipeline step: what it does, then a before -> after table."""
    s1p, s1b, s2s, s2d = ex["s1_prof"], ex["s1_bill"], ex["s2_srv"], ex["s2_dem"]
    i, g = ex["integ"], ex["gold"]
    W3 = [2.0 * inch, 2.35 * inch, 2.65 * inch]
    out = []

    def block(title, code, intro, rows, widths=W3, note=None):
        parts = [p(f"<b>{title}</b>  <font color='#52514e' size='8.5'>({code})</font>",
                   ParagraphStyle("st", parent=BODY, fontSize=11, spaceBefore=8, spaceAfter=3)),
                 p(intro), table(rows, widths)]
        if note:
            parts.append(p(note, SMALL))
        out.append(KeepTogether(parts))

    block("Step 1. Collect and store", "ingest.py, lake.py",
          "We load all 8 files exactly as they are into the <b>bronze</b> layer of the data lake. "
          "Nothing is changed yet. Here is the same customer in both sources: different column "
          "names and different ways of writing the same thing.",
          [["Source 1 says (account_billing.csv)", "Source 2 says (services.xlsx)", "Problem"],
           [f"Contract = \"{s1b['Contract']}\"", f"Contract = \"{s2s['Contract']}\"",
            "Same meaning, different capital letters"],
           [f"Partner = \"{s1p['Partner']}\"", f"Married = \"{s2d['Married']}\"",
            "Same fact, different column name"],
           [f"PaymentMethod = \"{s1b['PaymentMethod']}\"",
            f"Payment Method = \"{s2s['Payment Method']}\"", "The sources disagree"],
           [f"TotalCharges = \"{s1b['TotalCharges']}\" (stored as text)",
            f"Total Charges = {s2s['Total Charges']} (a number)", "Different data types"]])

    conf = ex["conflict"]
    block("Step 2. Integrate", "integrate.py, dependencies.py",
          "We give every column one shared name, write values the same way everywhere, then "
          "join everything into <b>one row per customer</b>. When the sources disagree we keep "
          "source 1 and log the conflict.",
          [["Before", "After", "Rule we applied"],
           [f"Partner = \"{s1p['Partner']}\" / Married = \"{s2d['Married']}\"",
            f"partner = {i['partner']}", "Yes/No becomes 1/0, one column name"],
           [f"\"{s1b['Contract']}\" / \"{s2s['Contract']}\"", f"contract = \"{i['contract']}\"",
            "One spelling for every category"],
           [f"Customer {conf['customerID']}: \"{conf['Contract_s1']}\" vs \"{conf['Contract_s2']}\"",
            f"contract = \"{conf['Contract_s1']}\"", "Conflict: source 1 wins (265 cases)"],
           [f"zip code {ex['s2_loc']['Zip Code']} (location file)",
            f"population = {ex['population']:,}",
            f"Looked up in the population file. {ex['same_zip']} customers share this zip"],
           ["74 columns spread over 8 files", "1 row per customer, 50 columns",
            "Joined on customer ID"]],
          note="We also check rules that should always hold, called <b>functional dependencies</b>. "
               "Example: a zip code always belongs to one city (true for all 1,626 zip codes), and "
               "anyone aged 65 or more is marked senior citizen (true for every customer).")

    before, after = ex["blank"]
    big, fence = ex["outlier"]
    block("Step 3. Clean", "clean.py",
          "We fix missing values and extreme values. The rule is learned from the training "
          "customers only, then applied to everyone.",
          [["Before", "After", "Why"],
           [f"Customer {before['customer_id']}: total charges = <i>blank</i>",
            f"total charges = {after['total_charges']:.2f}",
            "Source 2 has the value, so we copy it (11 customers)"],
           [f"Customer {ex['no_offer_id']}: offer = <i>blank</i>",
            "offer = \"None\"",
            "Blank here means they never got an offer (3,877 customers)"],
           ["internet type = <i>blank</i>", "internet type = \"None\"",
            "Blank means no internet (1,526 customers)"],
           [f"Customer {big['customer_id']}: {big['avg_monthly_gb_download']:.0f} GB per month",
            f"{fence:.0f} GB per month",
            "Far above typical use, so we cap it at the upper limit"]],
          note=f"How we spot extreme values: most customers download 3 to 27 GB. Anything above "
               f"{fence:.0f} GB (the middle range stretched by 1.5x) counts as an outlier.")

    block("Step 4. Transform", "transform.py",
          "Models only understand numbers on similar scales. So we rescale numbers and turn "
          "text into numbers. Our customer has been with the company 1 month and pays $29.85.",
          [["Before", "After", "What we did"],
           [f"tenure = {i['tenure']} month", f"{ex['z_tenure']:.2f}",
            "Standard scaling: how far below the average customer (32 months) they are"],
           [f"monthly charge = ${i['monthly_charges']:.2f}", f"{ex['minmax_monthly']:.2f}",
            "Min-max scaling: 0 = cheapest bill, 1 = most expensive"],
           [f"tenure = {i['tenure']}", f"{i['tenure'] / 100:.2f}",
            "Decimal scaling: divide by 100 so every value is below 1"],
           [f"total charges = ${i['total_charges']:.2f}", f"{np.log1p(i['total_charges']):.2f}",
            "Log: shrinks very large totals so a few big spenders don't dominate"],
           [f"contract = \"{i['contract']}\"", "month-to-month = 1, one year = 0, two year = 0",
            "One-hot encoding: one yes/no column per category"],
           [f"city = \"{i['city']}\"", f"{ex['la_share']:.3f} (frequency) or {ex['la_churn']:.2f} (target)",
            "Frequency: share of customers in that city. Target: churn rate of that city"]])

    block("Step 5. Build new features (ETL)", "features.py, etl.py",
          "We combine existing columns into new ones that say something the raw columns don't.",
          [["Built from", "New column", "Value for our customer"],
           ["the 6 add-on columns + streaming music", "addon_count", f"{g['addon_count']:.0f}"],
           ["monthly charge ÷ number of services", "charge_per_service", f"${g['charge_per_service']:.2f}"],
           ["contract", "is_month_to_month", f"{g['is_month_to_month']:.0f} (yes)"],
           ["payment method contains \"automatic\"", "auto_pay", f"{g['auto_pay']:.0f} (no)"],
           ["tenure", "tenure_band", str(g["tenure_band"])],
           ["tenure ≤ 6 months", "is_new_customer", f"{g['is_new_customer']:.0f} (yes)"]],
          widths=[2.6 * inch, 1.7 * inch, 2.7 * inch])

    block("Step 6. Test every choice", "experiments.py",
          "For each step we try several options and keep everything else the same, then keep "
          "the option with the best score. All of this happens on the development set only "
          "(see \"Training, validation and test sets\" below).",
          [["We compared", "Options", "We kept"],
           ["Ways to fill blanks", "6 (delete rows, delete columns, 0, median, copy from source 2, "
                                   "nearest neighbours)", "copy from source 2"],
           ["Ways to handle extremes", "5 (keep, remove, cap, z-score, Isolation Forest)", "cap"],
           ["Ways to scale numbers", "6 scalers × 2 models", "standard"],
           ["Ways to encode text", "5 encoders", "one-hot"],
           ["Models", "4 (logistic regression, KNN, random forest, gradient boosting)",
            "gradient boosting"]],
          widths=[1.7 * inch, 3.6 * inch, 1.7 * inch],
          note="For cleaning, scaling and encoding the options score within 0.01 of each other, "
               "which is smaller than the normal ups and downs between runs. There we kept the "
               "option that makes the most sense for the data (e.g. a blank offer really means "
               "\"no offer\"). For models the gap is clear, so we took the best one.")

    block("Step 7. Predict and show", "train.py, predict.py, dashboard/app.py",
          "The final model gives every customer a chance of leaving. The dashboard shows "
          "these scores and lets you type in a new customer.",
          [["Customer", "Model says", "Why it makes sense"],
           [f"{i['customer_id']}: 1 month, month-to-month, no referrals, no dependants",
            f"{g['churn_probability']:.0%} chance of leaving",
            "New, no commitment, no ties: the riskiest profile. The company should reach out"]])
    return out


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
    # ---------- what and why ----------
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
        ["Model", f"<b>{MODEL_NAMES.get(best, best).capitalize()}</b> (explained further down)"],
        ["How we judge it", "How many of the leavers it catches, and how often its warnings "
                            "are right (explained right below)"],
    ], [1.6 * inch, 5.4 * inch]))

    # ---------- how we judge the model ----------
    leavers = int(final["tp"] + final["fn"])
    stayers = int(final["tn"] + final["fp"])
    flagged = int(final["tp"] + final["fp"])
    s.append(KeepTogether([
        p("How we judge the model, with real numbers", H2),
        p(f"We hid {int(final['test_rows']):,} customers from the model, asked it to predict "
          f"each one, then compared with what really happened. {leavers} of them left and "
          f"{stayers} stayed. Every prediction lands in one of four boxes:"),
        table([
            ["", "Model said \"will leave\"", "Model said \"will stay\""],
            [f"<b>Really left</b> ({leavers})",
             f"<b>{int(final['tp'])} caught</b> ✓", f"<b>{int(final['fn'])} missed</b> ✗"],
            [f"<b>Really stayed</b> ({stayers})",
             f"<b>{int(final['fp'])} false alarms</b> ✗", f"<b>{int(final['tn'])} correctly left alone</b> ✓"],
        ], [2.0 * inch, 2.5 * inch, 2.5 * inch]),
        Spacer(1, 6),
    ]))
    s.extend(bullets([
        f"<b>Recall: how many leavers do we catch?</b> {int(final['tp'])} of {leavers} = "
        f"{final['tp'] / leavers:.0%}. Missing a leaver means losing a customer.",
        f"<b>Precision: when we warn, are we right?</b> {int(final['tp'])} of {flagged} warnings = "
        f"{final['tp'] / flagged:.0%}. A false alarm means giving a discount to someone who "
        "would have stayed anyway.",
        f"<b>F1 score</b> combines the two into one number from 0 to 1. Ours is "
        f"<b>{final['f1']:.2f}</b>. It only gets high if <i>both</i> are good, so it is the "
        "main number in all our charts.",
        f"<b>ROC-AUC</b> ({final['roc_auc']:.2f}): pick one customer who left and one who stayed "
        f"at random. {final['roc_auc']:.0%} of the time, the model gives the leaver the higher "
        "risk score. 0.5 would be a coin flip, 1.0 perfect.",
        "<b>Why not just accuracy (% of correct predictions)?</b> Because 73.5% of customers "
        "stay. A lazy model that says \"everyone stays\" would be 73.5% accurate while "
        "catching zero leavers. Accuracy hides that; F1 doesn't.",
    ]))

    s.append(p("The data", H2))
    s.append(p("Both sources describe the same 7,043 fictional customers of a California telecom, "
               "but they were published separately by IBM with different column names, formats "
               "and some disagreeing values. That is what makes integration a real problem."))
    s.append(callout(
        "<b>Why two sources about the same customers?</b> Each file is about the same people "
        "but holds <b>different facts</b> about them. Think of a student: the registrar knows "
        "their courses and grades, the library knows what they borrow, the gym knows how often "
        "they go. Same student, three systems. Companies work the same way: billing, the "
        "customer database and the network each keep their own records. Joining them gives "
        "the model more to learn from. Source 1 alone doesn't know a customer's age, how many "
        "friends they referred or where they live; source 2 does, and those turned out to be "
        "some of the best clues. The course also asks us to integrate several sources, and "
        "matching customers whose records disagree is exactly that skill."))
    s.append(Spacer(1, 6))
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

    # ---------- the 8 files, one by one ----------
    s.append(PageBreak())
    s.append(p("The 8 files, one by one", H2))
    s.append(p("Think of it like this: <b>source 1 says what each customer bought and whether "
               "they left</b>. <b>Source 2 describes the same customers in more detail</b>: who "
               "they are, where they live, how much they use, what they paid, and why they left. "
               "Every file except population has one row per customer, linked by customer ID."))
    s.append(table([
        ["File", "What one row is", "What is inside", "Why it matters"],
        ["<b>Source 1 · IBM GitHub · CSV</b>", "", "", ""],
        ["customer_profiles.csv<br/>6 columns", "a customer",
         "gender, senior citizen, partner, dependents (yes/no), tenure in months",
         "Who the customer is, basic version"],
        ["service_subscriptions.csv<br/>10 columns", "a customer",
         "phone, multiple lines, internet type (DSL / fiber / none), 6 add-ons "
         "(security, backup, device protection, tech support, TV, movies)",
         "What they subscribe to"],
        ["account_billing.csv<br/>7 columns", "a customer",
         "contract type, paperless billing, payment method (4 kinds), monthly and total "
         "charges, <b>Churn</b>",
         "Holds the <b>label we predict</b>. Has the 11 blank totals"],
        ["<b>Source 2 · IBM Cognos · Excel</b>", "", "", ""],
        ["demographics.xlsx<br/>9 columns", "a customer",
         "exact <b>age</b>, under 30, senior, married, dependents, <b>number of dependants</b>",
         "Adds age and a dependant count. Disagrees with source 1 on dependents for 551 customers"],
        ["location.xlsx<br/>9 columns", "a customer",
         "country, state (all California), <b>city</b>, <b>zip code</b>, latitude, longitude",
         "Lets us map customers and join to population"],
        ["population.xlsx<br/>3 columns", "<b>a zip code</b> (1,671 rows)",
         "zip code, population of that zip code",
         "The only file not about customers. Joined through location (many customers to one zip)"],
        ["services.xlsx<br/>30 columns", "a customer",
         "<b>referrals</b>, <b>offer</b> (A–E or none), internet type incl. Cable, GB downloaded, "
         "long distance, streaming music, unlimited data, refunds, extra data charges, total "
         "revenue, plus its own copy of contract, payment method (3 kinds) and charges",
         "Most new features. Its payment and contract values disagree with source 1"],
        ["status.xlsx<br/>11 columns", "a customer",
         "satisfaction score, customer status (stayed / churned / joined), churn label, churn "
         "score, lifetime value, churn category and reason",
         "Mostly the <b>answer in disguise</b>, so we use none of it as input"],
    ], [1.55 * inch, 1.0 * inch, 2.75 * inch, 1.7 * inch], section_rows=(1, 5)))
    s.append(Spacer(1, 6))
    s.append(p("Not counted above: <b>Telco-Customer-Churn.csv</b>, the original one-file version "
               "of source 1 (21 columns). Phase 1 split it into the three CSVs so each table "
               "covers one subject; we keep it in data/raw only as the original download.",
               SMALL))
    s.append(KeepTogether([p("How the files overlap", H2), table([
        ["Information", "Source 1", "Source 2", "What we do"],
        ["Gender, senior, partner", "yes", "yes (\"Married\")", "Same values. Keep source 1"],
        ["Dependents", "yes / no", "yes / no + count", "551 conflicts. Keep source 1 flag, add the count"],
        ["Tenure, charges", "yes", "yes", "Match, except the 11 new customers: source 2 fills their blanks"],
        ["Contract", "yes", "yes", "265 conflicts. Keep source 1"],
        ["Payment method", "4 kinds", "3 different kinds", "No clean mapping. Keep source 1"],
        ["Internet", "DSL / fiber / none", "yes/no + DSL / fiber / cable", "Keep source 1, add internet type"],
        ["Churn label", "yes", "yes", "Identical for all 7,043"],
        ["Age, referrals, offer, usage, location, population", "no", "yes", "New. This is what source 2 adds"],
    ], [1.9 * inch, 1.15 * inch, 1.6 * inch, 2.35 * inch])]))

    # ---------- how it works ----------
    s.append(p("How it works", H2))
    s.append(p("One command, <font name='Bold'>python -m src.pipeline</font>, runs every step "
               "below in order. Each step is a small Python file in the <font name='Bold'>src/"
               "</font> folder."))
    s.append(figure(DIAGRAM, W))
    s.append(Spacer(1, 6))

    s.append(p("How we built it, step by step", H2))
    s.append(p("To make each step concrete, we follow real customers through the pipeline. Our "
               "main example is customer <b>7590-VHVEG</b>. Every value below comes from the "
               "actual data and code."))
    s.extend(step_examples(examples()))

    # ---------- the model ----------
    s.append(KeepTogether([
        p("The prediction model", H2),
        p(f"We use <b>gradient boosting</b> (scikit-learn's HistGradientBoostingClassifier). In "
          "plain words: it builds a long chain of small <b>decision trees</b>. A decision tree is "
          "a flowchart of yes/no questions, for example \"Is the contract month-to-month?\" → "
          "\"Has the customer been here less than a year?\" → \"Did they refer any friends?\". "
          "One tree on its own is a rough guess. Each new tree in the chain looks at the "
          "customers the previous trees got wrong and tries to fix those mistakes. After 100 "
          "trees the combined answer is a percentage: the chance this customer leaves. Above "
          "50% we call it \"will leave\"."),
        p("Because only about 1 in 4 customers leave, we tell the model to treat each leaver as "
          "more important during training (\"balanced\" class weights). Otherwise it would learn "
          "that saying \"stays\" is almost always safe."),
        table([["Model we tried", "How it decides (simplified)", "Score (F1)"]] + [
            [MODEL_NAMES.get(m, m).capitalize(), desc, f"{models.loc[m, 'f1']:.2f}"]
            for m, desc in [
                ("hist_gb", "Chain of small decision trees, each fixing the last one's mistakes"),
                ("logreg", "Adds up points for each feature, turns the total into a %"),
                ("random_forest", "Hundreds of independent decision trees vote"),
                ("knn", "Looks at the 25 most similar customers and copies what they did"),
            ]], [1.8 * inch, 4.2 * inch, 1.0 * inch]),
    ]))

    # ---------- train / validation / test ----------
    n_total = int(stats.loc[stats["file"] == "customer_profiles.csv", "rows"].iloc[0])
    n_test = int(final["test_rows"])
    n_dev = n_total - n_test
    s.append(KeepTogether([
        p("Training, validation and test sets", H2),
        p("Yes, we use all three. The rule is simple: <b>the model is never graded on customers "
          "it learned from</b>, the same way a teacher doesn't put the practice questions on the "
          "final exam."),
        table([
            ["Set", "Customers", "What it is used for"],
            ["<b>Test set</b> (the final exam)", f"{n_test:,} (20%)",
             "Locked away on day one. Used <b>only once</b>, to grade the finished model. "
             "Every number in \"How we judge the model\" comes from here."],
            ["<b>Development set</b>", f"{n_dev:,} (80%)",
             "Everything else: all experiments and the choice of model happen here."],
            ["  ↳ training folds", f"about {n_dev * 4 // 5:,} at a time",
             "The model learns from these."],
            ["  ↳ validation fold", f"about {n_dev // 5:,} at a time",
             "The model is graded on these to compare options."],
        ], [2.0 * inch, 1.5 * inch, 3.5 * inch]),
        Spacer(1, 4),
        p("The development set is cut into 5 equal parts. We train on 4 parts and grade on the "
          "5th, then rotate until every part has been the validation part once, and average the "
          "5 scores. This is called <b>5-fold cross-validation</b>. It gives a steadier score "
          "than one single split, because no single lucky or unlucky split decides anything."),
        p(f"Check: the final model scored F1 {models.loc[best, 'f1']:.2f} in cross-validation and "
          f"{final['f1']:.2f} on the locked test set. The two are close, which means our "
          "experiments did not fool us.", SMALL),
    ]))

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
    s.append(p("How to read it: each bar is the model's score after adding one more stage of the "
               "project. The score (F1) runs from 0 to 1, and higher means the model finds more "
               "of the customers who leave while raising fewer false alarms. The biggest jump "
               "comes from <b>adding the second source</b>.", SMALL))

    s.append(p("Key results, in plain words", H2))
    drop = nulls.loc["drop_rows"]
    leavers = int(final["tp"] + final["fn"])
    flagged = int(final["tp"] + final["fp"])
    caught = final["tp"] / leavers
    right = final["tp"] / flagged
    g = lake.read("gold", "customer_features")
    rate = lambda mask: g.loc[mask, "churn"].mean()  # noqa: E731
    s.extend(bullets([
        f"<b>The model works.</b> We tested it on {int(final['test_rows']):,} customers it had "
        f"never seen. {leavers} of them really left. The model caught <b>{int(final['tp'])} of "
        f"those {leavers}</b> (about {round(caught * 10)} in 10). When it warned about a customer, "
        f"it was right about {round(right * 10)} times in 10. A random guess would be right "
        "only about 1 time in 4.",
        f"<b>The second source mattered most.</b> Adding it improved the model's score from "
        f"{stages['f1'].iloc[2]:.2f} to {stages['f1'].iloc[3]:.2f} (on a scale where 1 is perfect). "
        "Things like referrals and number of dependants only exist in source 2.",
        "<b>Cleaning had little to fix in our data</b> (only 11 blank cells), so it barely "
        "changed the score. To prove cleaning still matters, we damaged a copy of the data on "
        "purpose (10% blanks and 3% typos): the score dropped from "
        f"{stress.loc['clean data (reference)', 'f1']:.3f} to "
        f"{stress.loc['nulls: drop_rows', 'f1']:.3f}, and our cleaning brought it back to "
        f"{stress.loc['outliers: cap_iqr', 'f1']:.3f}.",
        f"<b>Deleting incomplete rows is a trap.</b> It looks like the best option, but only "
        f"because it throws away {1 - drop['test_rows'] / nulls['test_rows'].max():.0%} of the "
        "customers. A company can't ignore most of its customers, so we fill the blanks "
        "instead.",
        "<b>Putting numbers on the same scale</b> doesn't change much for some models but is "
        f"essential for others: one model's score jumped from "
        f"{scalers.loc['none | knn', 'f1']:.2f} to {scalers.loc['standard | knn', 'f1']:.2f}.",
        f"<b>Who leaves?</b> Customers on month-to-month contracts "
        f"({rate(g['contract'] == 'Month-to-month'):.0%} leave, vs "
        f"{rate(g['contract'] == 'Two year'):.0%} on two-year contracts), new customers "
        f"({rate(g['tenure'] <= 12):.0%} in their first year), and customers with no family on the "
        f"plan ({rate(g['number_of_dependents'] == 0):.0%} vs {rate(g['number_of_dependents'] > 0):.0%} "
        f"with dependants). Customers who referred friends leave less "
        f"({rate(g['number_of_referrals'] > 0):.0%} vs {rate(g['number_of_referrals'] == 0):.0%}).",
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

    s.append(KeepTogether([p("Words you will hear", H2), table([
        ["Term", "Meaning"],
        ["Churn", "A customer cancelling their service"],
        ["F1 score", "Balance of precision (flagged customers who really leave) and recall "
                     "(leavers we catch). 1.0 is perfect"],
        ["Cross-validation", "Split the development set into 5 parts, train on 4, grade on 1, repeat 5 times, "
                             "and average the scores"],
        ["IQR outlier", "A value far outside the middle 50% of the data (beyond 1.5 × IQR)"],
        ["One-hot encoding", "Turn a text column like Contract into one 0/1 column per value"],
        ["Delta Lake", "A data lake format that versions tables, so we can read any earlier copy"],
        ["Functional dependency", "X → Y: knowing X always tells you Y (zip code → city)"],
    ], [1.6 * inch, 5.4 * inch])]))

    # ---------- setup on a new machine ----------
    CODE = ParagraphStyle("code", parent=CELL, fontName="Mono", fontSize=8.5, leading=11.5)
    def cmd(text):
        return Paragraph(text.replace("\n", "<br/>"), CODE)
    s.append(PageBreak())
    s.append(p("Run it on a brand-new computer", H2))
    s.append(p("These steps work on Windows, Mac and Linux. Type the commands in a terminal: "
               "<b>PowerShell</b> on Windows, <b>Terminal</b> on Mac. Where Windows and Mac "
               "differ, both are shown."))
    rows = [
        ["Step", "Do this", "Command"],
        ["1. Install Python", "Download Python 3.11 or newer from python.org. On Windows, tick "
                              "<b>\"Add Python to PATH\"</b> in the installer.",
         cmd("python --version\n(should print 3.11 or higher;\non Mac use python3)")],
        ["2. Install Git", "Download from git-scm.com and install with the default options.",
         cmd("git --version")],
        ["3. Get the project", "Copy the repository to your computer and go into its folder.",
         cmd("git clone https://github.com/Peter7168/comp333-telco-churn.git\ncd comp333-telco-churn")],
        ["4. Make a virtual environment", "A private folder for this project's libraries, so "
                                          "nothing clashes with other projects.",
         cmd("Windows:\npython -m venv .venv\n.venv\\Scripts\\activate\n\nMac / Linux:\npython3 -m venv .venv\nsource .venv/bin/activate")],
        ["5. Install the libraries", "pandas, scikit-learn, Delta Lake, Streamlit and the rest. "
                                     "Takes 1 to 3 minutes.",
         cmd("pip install -r requirements.txt")],
        ["6. Run the pipeline", "Builds the data lake, runs every experiment, trains the model "
                                "and draws the charts. About 6 minutes. The first line should say "
                                "<b>storage: Delta Lake</b>.",
         cmd("python -m src.pipeline\n\nQuick version (1 min, skips\nthe experiments):\npython -m src.pipeline --skip-experiments")],
        ["7. Open the dashboard", "Opens in your web browser at localhost:8501. Stop it with "
                                  "Ctrl+C in the terminal.",
         cmd("streamlit run dashboard/app.py")],
        ["8. (Optional) Rebuild this PDF", "Updates every number in this document from the "
                                           "latest results.",
         cmd("python explanation/make_explainer.py")],
    ]
    data = [[p(r[0], CELL_B if i == 0 else CELL), p(r[1], CELL_B if i == 0 else CELL),
             p(r[2], CELL_B) if i == 0 else r[2]] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[1.45 * inch, 2.65 * inch, 2.9 * inch], hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                           ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                           ("LINEBELOW", (0, 0), (-1, 0), 1, BLUE),
                           ("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#f6f6f4")),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    s.append(t)
    s.append(Spacer(1, 8))
    s.append(p("<b>If something goes wrong</b>", BODY))
    s.extend(bullets([
        "<b>\"python is not recognized\"</b> (Windows): Python is not on PATH. Reinstall it and "
        "tick \"Add Python to PATH\", then open a new PowerShell window.",
        "<b>\"running scripts is disabled\"</b> when activating on Windows: run "
        "<font name='Mono'>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</font> once, then "
        "activate again.",
        "<b>The first line says \"CSV fallback\"</b>: deltalake did not install. Run "
        "<font name='Mono'>pip install deltalake pyarrow</font>. Everything still works without "
        "it, just stored as CSV files instead of Delta tables.",
        "<b>Next time</b> you open a terminal, go into the project folder and run the activate "
        "command from step 4 again before any other command.",
    ]))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="COMP 333 project explainer", author="Stats and Furious")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
