#!/usr/bin/env bash
# Rebuild the full forecasting pipeline from raw parquet -> metrics table.
# Usage: bash scripts/rebuild_all.sh
# Requires Python venv already activated and OptionMetrics parquet files present.
set -euo pipefail

echo "[1/6]  Building SPX price features"
python -m econ499.feats.build_price | cat

echo "[2/6]  Building IV-surface statistics"
python -m econ499.feats.build_iv_surface | cat

echo "[3/6]  Building IV FPCA factors"
python -m econ499.feats.build_iv_fpca | cat

echo "[4/6]  Building macro features"
python -m econ499.feats.fetch_macro | cat

echo "[5/6]  Merging feature sets"
python -m econ499.panels.merge_features | cat

# ------------------- hyper-parameter tuning -------------------

echo "[TUNE]  Running Optuna hyper-parameter searches (LSTM, PPO, A2C)"
# -- Optuna HPO (CLI flags differ slightly across modules)
python -m econ499.tune.hpo_lstm --trials 30 | cat
python -m econ499.tune.hpo_ppo --n-trials 30 | cat
python -m econ499.tune.hpo_a2c --n-trials 30 | cat

# ------------------- baselines -------------------

DATA_DIR="$(python - <<PY
import yaml, pathlib, json, sys
cfg = yaml.safe_load(open('cfg/data_config.yaml'))
print(pathlib.Path(cfg['paths']['output_dir']).resolve())
PY)"

echo "[6/6]  Training / running baselines"
python -m econ499.benchmarks.har_rv | cat
python -m econ499.benchmarks.ridge | cat
python -m econ499.benchmarks.garch | cat
python -m econ499.benchmarks.ols | cat
python -m econ499.benchmarks.lstm --param_file "$DATA_DIR/best_lstm_params.json" || true

# ------------------- DRL -------------------

echo "[TRAIN] Training PPO/A2C with λ ∈ {0,10,20} (this may take a while)"
for L in 0 10 20; do
  echo "  → PPO λ=$L"
  python -m econ499.agents.ppo --timesteps 500000 --arb_lambda "$L" | cat
  echo "  → A2C λ=$L"
  python -m econ499.agents.a2c --timesteps 500000 --arb_lambda "$L" | cat

  # Generate out-of-sample forecasts for each model so evaluator picks them up
  for ALG in ppo a2c; do
    if [[ "$L" -eq 0 ]]; then
      MODEL_DIR="$DATA_DIR/${ALG}_best_model"
    else
      MODEL_DIR="$DATA_DIR/${ALG}_best_model_l$L"
    fi
    if [[ -d "$MODEL_DIR" ]]; then
      OUT_CSV="$DATA_DIR/${ALG}_l${L}_oos_predictions.csv"
      python -m econ499.predict.make_drl_forecast --model "$MODEL_DIR" --suffix "l$L" --out "$OUT_CSV" | cat
    fi
  done
done

# ------------------- evaluation -------------------

echo "[EVAL] Running evaluator (suppressing runpy warning)"
python -W ignore::RuntimeWarning -m econ499.eval.evaluate_all --dm_base har_rv --mcs --mcs_alpha 0.1 | cat

# Save tuned params snapshot for reproducibility
python scripts/snapshot_best_params.py | cat

# ------------------- done -------------------

echo "Full rebuild complete. Metrics table -> artifacts/tables/forecast_metrics.csv" 