import os
from pathlib import Path

from econ499.baselines.ar1 import run_ar1


def test_run_ar1(tmp_path):
    out_csv = tmp_path / "ar1_out.csv"
    path = run_ar1(out_csv=out_csv, train_ratio=0.9)
    assert Path(path).exists()
    data = Path(path).read_text().splitlines()
    assert len(data) > 1
