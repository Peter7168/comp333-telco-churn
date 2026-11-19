"""Delta Lake storage (bronze = raw, silver = integrated/clean, gold = features).

Uses the deltalake package; falls back to CSV if it is not installed.
"""
import sqlite3

import pandas as pd

from src.config import LAKE, WAREHOUSE

try:
    from deltalake import DeltaTable, write_deltalake
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False


def _path(layer, name):
    return LAKE / layer / name


def write(df, layer, name):
    """Save a dataframe as a table in one layer of the lake."""
    path = _path(layer, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_DELTA:
        # Overwrite creates a new Delta version each run.
        write_deltalake(str(path), df, mode="overwrite", schema_mode="overwrite")
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)
    return path


def read(layer, name, version=None):
    """Read a table. With Delta, version=0, 1, ... reads an older copy (time travel)."""
    path = _path(layer, name)
    if HAS_DELTA and path.exists():
        return DeltaTable(str(path), version=version).to_pandas()
    # Only empty cells are missing (keeps the text "None").
    return pd.read_csv(path.with_suffix(".csv"), keep_default_na=False, na_values=[""])


def history(layer, name):
    """Version log of a Delta table (empty when Delta is not installed)."""
    if not HAS_DELTA:
        return []
    return DeltaTable(str(_path(layer, name))).history()


def to_warehouse(df, name):
    """Copy a table into SQLite so the dashboard can query it with SQL."""
    with sqlite3.connect(WAREHOUSE) as con:
        df.to_sql(name, con, if_exists="replace", index=False)


def query(sql):
    with sqlite3.connect(WAREHOUSE) as con:
        return pd.read_sql_query(sql, con)
