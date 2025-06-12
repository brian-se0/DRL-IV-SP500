#!/usr/bin/env bash
# Run DRL ablation study by excluding each feature block (surface, realised, macro) one at a time.
set -euo pipefail

# Resolve absolute data_processed directory from YAML so paths work regardless of CWD
DATA_DIR="$(python - <<PY
import yaml, pathlib, sys; cfg=yaml.safe_load(open('data_config.yaml')); print(pathlib.Path(cfg['paths']['output_dir']).resolve())
PY)"

BLOCKS=(surface realised macro)
TIMESTEPS=250000  # per-agent training budget for each ablation

MODEL_PPO_DIR="$DATA_DIR/ppo_best_model_l10"
MODEL_A2C_DIR="$DATA_DIR/a2c_best_model_l10"
MODEL_PPO="$MODEL_PPO_DIR/best_model.zip"
MODEL_A2C="$MODEL_A2C_DIR/best_model.zip"

for block in "${BLOCKS[@]}"; do
  echo "[ABLATION] Training PPO without block: $block"
  python -m econ499.agents.ppo --timesteps "$TIMESTEPS" --exclude_block "$block" --arb_lambda 10 | cat

  echo "[ABLATION] Saving PPO forecasts"
  python -m econ499.predict.make_drl_forecast \
         --model "$MODEL_PPO" \
         --exclude_block "$block" \
         --out    "$DATA_DIR/ppo_${block}_oos_predictions.csv" \
         | cat

  echo "[ABLATION] Training A2C without block: $block"
  python -m econ499.agents.a2c --timesteps "$TIMESTEPS" --exclude_block "$block" --arb_lambda 10 | cat

  echo "[ABLATION] Saving A2C forecasts"
  python -m econ499.predict.make_drl_forecast \
         --model "$MODEL_A2C" \
         --exclude_block "$block" \
         --out    "$DATA_DIR/a2c_${block}_oos_predictions.csv" \
         | cat

done

# Aggregate metrics for ablation runs (including new forecasts)
python -m econ499.evaluation.evaluate_all --dm_base har_rv --mcs --mcs_alpha 0.10 | cat

echo "Ablation study complete. Updated metrics available at artifacts/tables/forecast_metrics.csv" 