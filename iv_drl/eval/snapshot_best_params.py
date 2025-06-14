from __future__ import annotations

"""Save a snapshot of the best hyper-parameters for each model."""

import json
import logging
from pathlib import Path
import shutil
import yaml

from iv_drl.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config("data_config.yaml")

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / 'config/data_config.yaml'))
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