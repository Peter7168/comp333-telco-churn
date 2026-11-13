"""Run the whole project end to end:  python -m src.pipeline

  1. ingest       raw CSV + XLSX files      -> bronze
  2. integrate    schema mapping + joins    -> silver/customers_integrated
  2b. entity resolution: match the sources without the ID, as a check on the join
  3. dependencies functional dependency checks
  4. etl          cleaning + new features   -> silver/customers_clean, gold/customer_features
  5. experiments  F1 with and without each cleaning / transformation step
  6. train        final model               -> models/churn_model.joblib
  7. figures      charts for the slides     -> results/figures

Add --skip-experiments to skip step 5 (it takes about 5 minutes).
"""
import sys
import time

from src import (dependencies, entity_resolution, etl, experiments, figures, ingest,
                 integrate, lake, train)


def step(name, fn, *args):
    start = time.time()
    print(f"\n=== {name} ===")
    result = fn(*args)
    print(f"--- {name} done in {time.time() - start:.1f}s")
    return result


def main():
    skip_experiments = "--skip-experiments" in sys.argv
    print(f"storage: {'Delta Lake' if lake.HAS_DELTA else 'CSV fallback (pip install deltalake)'}")

    step("ingest", ingest.run)
    integrated = step("integrate", integrate.run)
    step("entity resolution", entity_resolution.run)
    step("dependencies", dependencies.run, integrated)
    step("etl", etl.run, integrated)
    if not skip_experiments:
        step("experiments", experiments.run, integrated)
    step("train", train.run, integrated)
    step("figures", figures.run)


if __name__ == "__main__":
    main()
