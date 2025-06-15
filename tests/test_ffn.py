from pathlib import Path

from econ499.baselines.ffn import run_ffn


def test_run_ffn(tmp_path):
    out_csv = tmp_path / "ffn_out.csv"
    path = run_ffn(out_csv=out_csv, max_epochs=1)
    assert Path(path).exists()
    data = Path(path).read_text().splitlines()
    assert len(data) > 1
