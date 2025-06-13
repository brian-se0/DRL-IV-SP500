# Run all baseline models
$DATA_DIR = (python -c "import yaml, pathlib, sys; cfg=yaml.safe_load(open('config/data_config.yaml')); print(pathlib.Path(cfg['paths']['output_dir']).resolve())")

Write-Host "[BASE] Training/running econometric baselines"

Write-Host "Running HAR-RV baseline..."
python iv_drl/baselines/har_rv.py

Write-Host "Running Ridge regression baseline..."
python iv_drl/baselines/ridge.py

Write-Host "Running GARCH baseline..."
python iv_drl/baselines/garch.py

Write-Host "Running OLS baseline..."
python iv_drl/baselines/ols.py

Write-Host "Running LSTM baseline..."
python iv_drl/baselines/lstm.py --param_file "$DATA_DIR/best_lstm_params.json" 2>$null

Write-Host "Baseline models complete. Results saved in results/" 