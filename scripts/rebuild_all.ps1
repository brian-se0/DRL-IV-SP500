# Requires an activated venv and Python 3.12
Write-Host "[1/6]  Building SPX price features"
python build_spx_price_features.py

Write-Host "[2/6]  Building IV-surface statistics"
python build_iv_surface_features.py

Write-Host "[3/6]  Building IV FPCA factors"
python build_iv_fpca_factors.py

Write-Host "[4/6]  Building macro features"
python fetch_macro_series.py

Write-Host "[5/6]  Merging feature sets"
python merge_feature_sets.py

# ------------------- Hyper-parameter tuning -------------------
Write-Host "[TUNE] Running Optuna hyper-parameter searches (LSTM, PPO, A2C)"
python -m econ499.tune.hpo_lstm --trials 30
python -m econ499.tune.hpo_ppo --n-trials 30
python -m econ499.tune.hpo_a2c --n-trials 30

# ------------------- Baselines -------------------
$DATA_DIR = (python -c "import yaml, pathlib, sys; cfg=yaml.safe_load(open('data_config.yaml')); print(pathlib.Path(cfg['paths']['output_dir']).resolve())")

Write-Host "[BASE] Training/running econometric baselines"
python -m econ499.benchmarks.har_rv
python -m econ499.benchmarks.ridge
python -m econ499.benchmarks.garch
python -m econ499.benchmarks.ols
python -m econ499.benchmarks.lstm --param_file "$DATA_DIR/best_lstm_params.json" 2>$null

# ------------------- DRL -------------------
Write-Host "[TRAIN] Training PPO (this may take a while)"
python -m econ499.agents.ppo --timesteps 500000 --arb_lambda 10

Write-Host "[TRAIN] Training A2C (this may take a while)"
python -m econ499.agents.a2c --timesteps 500000 --arb_lambda 10

# ------------------- Evaluation -------------------
Write-Host "[EVAL] Running evaluator"
python -m econ499.evaluation.evaluate_all --dm_base har_rv --mcs --mcs_alpha 0.1

# Save tuned params snapshot
python scripts/snapshot_best_params.py

Write-Host "Full rebuild complete. Metrics table -> artifacts/tables/forecast_metrics.csv" 