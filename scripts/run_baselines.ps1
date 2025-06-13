# Run all baseline models
$DATA_DIR = (python -c "import yaml, pathlib, sys; cfg=yaml.safe_load(open('cfg/data_config.yaml')); print(pathlib.Path(cfg['paths']['output_dir']).resolve())")

Write-Host "[BASE] Training/running econometric baselines"

Write-Host "Running HAR-RV baseline..."
python econ499/baselines/har_rv.py

Write-Host "Running Ridge regression baseline..."
python econ499/baselines/ridge.py

Write-Host "Running GARCH baseline..."
python econ499/baselines/garch.py

Write-Host "Running OLS baseline..."
python econ499/baselines/ols.py

Write-Host "Running LSTM baseline..."
python econ499/baselines/lstm.py --param_file "$DATA_DIR/best_lstm_params.json" 2>$null

Write-Host "Baseline models complete. Results saved in results/" 