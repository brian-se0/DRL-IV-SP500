from __future__ import annotations

"""Thin driver that rebuilds all artifacts with a single command.

It simply calls the ``main`` / ``evaluate_all`` functions of each pipeline
stage in the correct order and logs progress.  Use ``python regenerate_artifacts.py``
or ``python -m econ499.orchestration.run_pipeline``.
"""

import argparse
import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from econ499.utils import load_config

CFG = load_config("data_config.yaml")
DATA_DIR = Path(CFG["paths"]["output_dir"]).resolve()
ARTIFACT_TABLE = Path(__file__).resolve().parents[2] / "artifacts" / "tables" / "forecast_metrics.csv"

# (module, callable, description, output path)
_TASKS: list[tuple[str, str, str, Path | None]] = [
    ("econ499.features.build_price", "main", "SPX price features", DATA_DIR / "spx_daily_features.parquet"),
    (
        "econ499.features.build_iv_surface",
        "main",
        "IV-surface features",
        DATA_DIR / "iv_surface_daily_features.parquet",
    ),
    ("econ499.features.fetch_macro", "main", "Macro series", DATA_DIR / "macro_daily_features.parquet"),
    ("econ499.panels.merge_features", "merge", "Merge feature sets", DATA_DIR / "spx_iv_drl_state.csv"),
    ("econ499.evaluation.evaluate_all", "evaluate_all", "Evaluate forecasts", ARTIFACT_TABLE),
]


def _import_callable(module_name: str, attr: str) -> Callable:  # pragma: no cover
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{module_name}.{attr} is not callable")
    return fn


def run_pipeline(force: bool = False) -> None:
    """Run every stage of the pipeline.

    Parameters
    ----------
    force : bool, default False
        If *True* existing output files will be deleted before running the
        corresponding stage.  This guarantees a full rebuild.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start = datetime.now()
    logging.info("Starting ECON-499 pipeline (force=%s)…", force)

    for module_name, attr, desc, output in _TASKS:
        try:
            if force and output is not None and output.exists():
                logging.info("-- force: deleting %s", output)
                output.unlink(missing_ok=True)
            logging.info('-> %s', desc)
            fn = _import_callable(module_name, attr)
            # Evaluate stage returns a Path; others may return None
            result = fn()  # type: ignore[arg-type]
            if isinstance(result, Path):
                logging.info('%s finished (%s)', desc, result)
            else:
                logging.info('%s finished', desc)
        except Exception as err:  # pragma: no cover
            logging.exception('FAILED: %s', desc)
            raise SystemExit(1) from err

    elapsed = datetime.now() - start
    logging.info("All stages completed in %s", elapsed)


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Rebuild all ECON-499 artifacts")
    ap.add_argument("--force", action="store_true", help="delete existing outputs before rebuilding")
    return ap.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    run_pipeline(force=args.force) 