"""Step 2b: entity resolution (ER) between the two sources.

Both sources carry a customer ID, so the join itself is easy. Here I check that
the ID link is trustworthy, and I show how the same customers can be matched
WITHOUT the ID, the way we would have to if a source had no shared key:

  1. key quality   - IDs are unique, well formed and match across sources
  2. duplicates    - is the same customer stored twice under two IDs?
  3. record linkage - blocking + attribute similarity, scored against the true IDs
"""
import pandas as pd

from src.config import TABLES
from src.integrate import build_source1, build_source2

# Blocking: only compare records that agree on these cheap, stable attributes.
BLOCK = ["gender", "senior_citizen", "partner", "phone_service", "paperless_billing",
         "monthly_charges"]
# Matching: attributes compared inside a block to score each candidate pair.
COMPARE = ["tenure", "total_charges", "contract", "dependents", "multiple_lines",
           "online_security", "online_backup", "device_protection", "tech_support",
           "streaming_tv", "streaming_movies"]
NUMERIC = ["tenure", "total_charges"]
ID_PATTERN = r"^\d{4}-[A-Z]{5}$"


def prepare(df):
    """Keep the comparison attributes and write them the same way in both sources."""
    out = df[["customer_id"] + BLOCK + COMPARE].copy()
    for col in COMPARE:
        if col not in NUMERIC and col != "dependents":
            # Source 1 says "No internet service" where source 2 just says "No".
            out[col] = out[col].replace({"No internet service": "No", "No phone service": "No"})
    out["monthly_charges"] = out["monthly_charges"].round(2)
    return out


def key_quality(s1, s2):
    ids1, ids2 = s1["customer_id"], s2["customer_id"]
    return {
        "source 1 IDs": len(ids1), "source 1 duplicate IDs": int(ids1.duplicated().sum()),
        "source 2 IDs": len(ids2), "source 2 duplicate IDs": int(ids2.duplicated().sum()),
        "IDs with the expected format (1234-ABCDE)":
            int(ids1.str.match(ID_PATTERN).sum() + ids2.str.match(ID_PATTERN).sum()),
        "IDs found in both sources": len(set(ids1) & set(ids2)),
    }


def duplicate_entities(df):
    """Customers that are identical on every compared attribute but have different IDs."""
    cols = BLOCK + COMPARE
    return int(df.duplicated(subset=cols, keep=False).sum())


def link_without_ids(s1, s2):
    """Match each source 1 record to a source 2 record using attributes only."""
    pairs = s1.merge(s2, on=BLOCK, suffixes=("_s1", "_s2"))

    # I score each candidate pair by the share of attributes that agree.
    agree = pd.DataFrame(index=pairs.index)
    for col in COMPARE:
        a, b = pairs[f"{col}_s1"], pairs[f"{col}_s2"]
        if col in NUMERIC:
            agree[col] = (a - b).abs().lt(0.01) | (a.isna() & b.isna())
        else:
            agree[col] = a.astype(str).eq(b.astype(str))
    pairs["score"] = agree.mean(axis=1)

    # The best-scoring candidate wins. A tie for first place means I cannot decide.
    best = pairs.groupby("customer_id_s1")["score"].transform("max")
    top = pairs[pairs["score"] == best]
    n_top = top.groupby("customer_id_s1")["customer_id_s2"].transform("size")
    decided = top[n_top == 1]
    correct = decided["customer_id_s1"].eq(decided["customer_id_s2"])

    n = len(s1)
    stats = {
        "all possible pairs (no blocking)": n * len(s2),
        "candidate pairs after blocking": len(pairs),
        "pairs avoided by blocking (%)": round(100 * (1 - len(pairs) / (n * len(s2))), 2),
        "true match kept in its block (%)":
            round(100 * pairs["customer_id_s1"].eq(pairs["customer_id_s2"]).sum() / n, 2),
        "customers matched to exactly one record": len(decided),
        "of those, matched to the right customer": int(correct.sum()),
        "precision (%)": round(100 * correct.mean(), 2),
        "recall (%)": round(100 * correct.sum() / n, 2),
        "ambiguous (tie between look-alike customers)": n - len(decided),
    }

    tied = top[n_top > 1].sort_values(["customer_id_s1", "customer_id_s2"])
    examples = tied[["customer_id_s1", "customer_id_s2", "monthly_charges", "tenure_s1",
                     "tenure_s2", "contract_s1", "contract_s2", "score"]].head(6)
    return stats, examples


def run():
    s1, s2 = prepare(build_source1()), prepare(build_source2())

    summary = key_quality(s1, s2)
    summary["source 1 look-alike customers (same attributes, different ID)"] = duplicate_entities(s1)
    summary["source 2 look-alike customers (same attributes, different ID)"] = duplicate_entities(s2)
    stats, examples = link_without_ids(s1, s2)
    summary.update(stats)

    # I store values as text so counts print as 7043, not 7043.00.
    values = [f"{v:,}" if isinstance(v, int) else str(v) for v in summary.values()]
    table = pd.DataFrame({"check": list(summary), "value": values})
    print(table.to_string(index=False))
    table.to_csv(TABLES / "entity_resolution.csv", index=False)
    examples.to_csv(TABLES / "entity_resolution_ambiguous_examples.csv", index=False)
    return table


if __name__ == "__main__":
    run()
