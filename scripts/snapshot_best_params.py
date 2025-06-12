from __future__ import annotations

"""Copy best hyper-parameter JSON files to a version-controlled `configs/` folder.

Run after `rebuild_all.sh` or tuning scripts so that the latest Optuna output is
snapshotted for reproducibility.

Usage:
    python scripts/snapshot_best_params.py
"""

import shutil
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / 'data_config.yaml'))
DATA_DIR = Path(CFG['paths']['output_dir']).resolve()
CONF_DIR = ROOT / 'configs'
CONF_DIR.mkdir(exist_ok=True)

for stem in ['best_lstm_params.json', 'best_ppo_params.json', 'best_a2c_params.json']:
    src = DATA_DIR / stem
    if src.exists():
        dst = CONF_DIR / stem
        shutil.copy2(src, dst)
        print('copied', src, '->', dst)
    else:
        print('WARN:', src, 'not found (skip)') 