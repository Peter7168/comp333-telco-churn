"""Small storage layer over Delta Lake.

The lake has three layers:
  bronze - raw files loaded as-is, one table per source file
  silver - integrated and cleaned customer table
  gold   - model-ready features

I use the `deltalake` package (delta-rs), so no Spark cluster is needed.
If it is not installed, I fall back to plain CSV in the same folders so the
pipeline still runs; the README says how to install it.
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
        # I overwrite so each pipeline run creates a new Delta version.
        write_deltalake(str(path), df, mode="overwrite", schema_mode="overwrite")
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)
    return path


def read(layer, name, version=None):
    """Read a table. With Delta, version=0, 1, ... reads an older copy (time travel)."""
    path = _path(layer, name)
    if HAS_DELTA and path.exists():
        return DeltaTable(str(path), version=version).to_pandas()
    # I only treat empty cells as missing. By default pandas also turns the text
    # "None" into a blank, which would undo our "no offer" fill.
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
