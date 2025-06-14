# A Deep Reinforcement Learning Approach to S&P 500 Implied Volatility Surface Forecasting

This repository implements a deep reinforcement learning framework for forecasting the S&P 500 implied volatility surface. The framework uses policy-gradient agents (PPO, A2C) to predict one-day-ahead ATM implied volatility while maintaining static arbitrage constraints.

## Motivation and Literature Review

### Implied Volatility Forecasting
The forecasting of implied volatility surfaces is a fundamental challenge in quantitative finance, with significant implications for option pricing, risk management, and trading strategies. Traditional approaches have relied on econometric models like GARCH [Bollerslev (1986)](https://doi.org/10.2307/1913246) and HAR-RV [Corsi (2009)](https://doi.org/10.1016/j.jeconom.2008.09.007), which make strong parametric assumptions about the data-generating process. Recent work by [Gonçalves & Guidolin (2022)](https://doi.org/10.1093/jjfinec/nbaa037) demonstrates the value of functional principal component analysis (FPCA) in capturing the dynamics of the volatility surface, which we incorporate into our feature engineering pipeline.

### Deep Reinforcement Learning in Finance
The application of deep reinforcement learning to financial forecasting has gained traction following the success of [Buehler et al. (2019)](https://doi.org/10.1016/j.jfineco.2019.11.002) in option pricing. Our framework builds on this work by extending DRL to volatility forecasting while maintaining no-arbitrage constraints, following the approach of [Buehler et al. (2023)](https://doi.org/10.1080/14697688.2023.2174567) in enforcing static arbitrage conditions through penalty terms.

### Feature Engineering
Our feature engineering approach is grounded in several key papers:
- Surface characteristics and term structure features follow [Carr & Wu (2009)](https://doi.org/10.1093/rfs/hhp009)
- Realized volatility measures incorporate the jump-robust estimators from [Zhang et al. (2005)](https://doi.org/10.1111/j.1467-9868.2005.00537.x)
- Macroeconomic indicators are selected based on [Liu et al. (2020)](https://doi.org/10.1016/j.jfineco.2020.03.008)
- FPCA decomposition follows [Cont & da Fonseca (2002)](https://doi.org/10.1016/S0378-4371(02)01565-3)

### Evaluation Framework
Our evaluation methodology incorporates:
- Diebold-Mariano tests [Diebold & Mariano (1995)](https://doi.org/10.2307/2291625)
- Model Confidence Set analysis [Hansen et al. (2011)](https://doi.org/10.1111/j.1467-9868.2010.00735.x)
- QLIKE loss function [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.07.013)

### Robustness Checks
The robustness of our results is assessed through:
- Feature block ablations following [Chen et al. (2023)](https://doi.org/10.1016/j.jeconom.2023.01.001)
- Alternative sample splits and walk-forward evaluation as in [Amaya et al. (2022)](https://doi.org/10.1016/j.ijforecast.2022.03.001)
- Static-arbitrage penalty sensitivity analysis based on [Buehler et al. (2023)](https://doi.org/10.1080/14697688.2023.2174567)

## Data and Feature Engineering

### Data Sources
- **OptionMetrics**: Daily implied volatility surfaces for S&P 500 index options (1996-2023)
- **High-Frequency Data**: 5-minute returns for realized volatility calculation
- **Macroeconomic Indicators**: GDP, inflation, unemployment, monetary policy indicators
- **Liquidity Proxy**: Amihud illiquidity uses SPY ETF dollar volume
- **Market Sentiment**: 
  - VIX and other volatility indices from FRED
  - VVIX (Cboe VIX of VIX Index) from [CBOE's historical data page](https://www.cboe.com/tradable_products/vix/vix_historical_data/)
  - For dates before VVIX data availability (2004-2006), we use a realized volatility of VIX (rvvix) as a proxy

### Feature Blocks
1. **Surface Characteristics**
   - ATM term structure (30, 60, 90, 180, 365 days)
   - Volatility skew at each maturity
   - Surface curvature and convexity measures

2. **Realized Volatility**
   - 5-minute realized volatility
   - Overnight returns volatility
   - Jump-robust volatility estimators

3. **Macroeconomic Indicators**
   - GDP growth and inflation expectations
   - Unemployment rate and labor market indicators
   - Monetary policy indicators
   - Market sentiment measures

4. **FPCA Decomposition**
   - First three principal components
   - Daily changes and momentum
   - Surface dynamics

## Methodology

### DRL Environment
- **State Space**: 70+ engineered features from all feature blocks
- **Action Space**: Vector of length ``n_maturities`` forecasting next-day ATM IV for each maturity
- **Reward Function**: Negative MSE with static-arbitrage penalty
- **Constraints**: No-arbitrage conditions enforced through penalty term

### Baselines
1. **Traditional Models**
   - HAR-RV
   - AR(1)
   - GARCH(1,1)

2. **Machine Learning**
   - LSTM
   - FFN (feed-forward network)
   - Ridge Regression
   - OLS

3. **Simple Benchmarks**
   - Naive (today's IV)
   - Rolling mean

To train the FFN baseline and save forecasts:
```bash
python -m econ499.baselines.ffn
```

## Implementation

### Environment Setup
```bash
# Create and activate virtual environment (Python 3.11 recommended)
python3.11 -m venv venv311
source venv311/bin/activate  # Linux/Mac
.\venv311\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0+cu121 torchvision torchaudio
```

### Running Tests
Install dev dependencies and run pytest to execute the test suite. Continuous integration should use the same commands.
```bash
pip install -r requirements-dev.txt
pytest
```


### Data Processing Pipeline
Set the directory containing the OptionMetrics daily ZIP files in
`cfg/data_config.yaml` under `paths.option_data_zip_dir`.
You must also set `paths.vvix_csv` to the path of the VVIX history CSV.
The data processing pipeline consists of the following scripts that must be run in this exact order. Each block can also be executed with a one-line PowerShell wrapper stored in `scripts/`:

```powershell
# 1. Feature Building
python -m econ499.feats.build_price
python -m scripts.process_option_data  # extracts OptionMetrics zip files
python -m econ499.feats.build_iv_surface  # applies liquidity filters and gap interpolation
python -m econ499.feats.build_iv_fpca
python -m econ499.feats.fetch_macro
python -m econ499.panels.merge_features
# Alternatively run all feature scripts at once
powershell -File scripts/run_features.ps1

# 2. Hyperparameter Optimization
python -m econ499.hpo.hpo_lstm --trials 30
python -m econ499.hpo.hpo_ppo --n-trials 30
python -m econ499.hpo.hpo_a2c --n-trials 30
python -m econ499.hpo.hpo_ffn --trials 30
# Alternatively run all HPO scripts at once
powershell -File scripts/run_hpo.ps1

# 3. Model Training
python -m econ499.baselines.lstm
python -m econ499.baselines.ffn
python -m econ499.models.ppo --timesteps 100000
python -m econ499.models.a2c --timesteps 100000
# Alternatively run all training scripts at once
powershell -File scripts/run_training.ps1

# Forecast multiple maturities (e.g. 30d and 90d)
# Set ``features.surface.maturities`` in ``cfg/data_config.yaml`` to ``[30, 90]``
# and run the same training commands. The agent will output a vector forecast
# with one prediction per maturity.

# 4. Evaluation
python -m econ499.eval.evaluate_all
python -m econ499.baselines.har_rv
python -m econ499.eval.evaluate_all --dm_base har_rv --mcs --mcs_alpha 0.1
python -m econ499.eval.make_ensemble results/lstm_oos_predictions.csv results/har_rv_oos_predictions.csv
python -m econ499.eval.eval_vix_regimes --vix_thresh 20
# Alternatively run the evaluation block at once
powershell -File scripts/run_evaluation.ps1
```

Note: Each script must be run in the order shown above to ensure all dependencies are available. Each script will save its outputs to the appropriate directories.

## Robustness Checks

All robustness experiments can be executed individually as shown below or all at once via:

```powershell
powershell -File scripts/run_robustness.ps1
```

### Feature Block Ablations
```powershell
# Train without macro features
python -m econ499.models.ppo --exclude_block macro
# The script will automatically copy the best model to a unique file, e.g.:
#   results/ppo_best_model/ppo_best_model_no_macro.zip

# Generate forecasts with a clear output name and column suffix
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_no_macro.zip --exclude_block macro --out results/ppo_no_macro_oos_predictions.csv --suffix no_macro
python -m econ499.models.a2c --exclude_block macro
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model.zip --exclude_block macro --out results/a2c_no_macro_oos_predictions.csv --suffix no_macro
```

### Static-Arbitrage Penalty Sensitivity
```powershell
# Train with no penalty
python -m econ499.models.ppo --arb_lambda 0
# The script will automatically copy the best model to a unique file, e.g.:
#   results/ppo_best_model/ppo_best_model_arb0.zip
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_arb0.zip --arb_lambda 0 --out results/ppo_arb0_oos_predictions.csv --suffix arb0
python -m econ499.models.a2c --arb_lambda 0
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_l0.zip --arb_lambda 0 --out results/a2c_arb0_oos_predictions.csv --suffix arb0

# Train with strong penalty
python -m econ499.models.ppo --arb_lambda 20
# The script will automatically copy the best model to a unique file, e.g.:
#   results/ppo_best_model/ppo_best_model_arb20.zip
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_arb20.zip --arb_lambda 20 --out results/ppo_arb20_oos_predictions.csv --suffix arb20
python -m econ499.models.a2c --arb_lambda 20
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_l20.zip --arb_lambda 20 --out results/a2c_arb20_oos_predictions.csv --suffix arb20
```

### Alternative Sample Splits
```powershell
# Walk-forward evaluation (output is always a new file)
python -m econ499.eval.eval_walk_forward --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_metrics_walk.csv

# Hold-out evaluation (output is always a new file)
python -m econ499.eval.eval_alt_splits --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_metrics_alt_splits.csv
```

### Multi-Seed Experiments
```powershell
# After running multiple seeds and generating files like ppo_seed42_oos_predictions.csv, etc.
python -m econ499.eval.eval_multi_seed --pattern "ppo_seed*_oos_predictions.csv" --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/seed_run_summary_ppo.csv
python -m econ499.eval.eval_multi_seed --pattern "a2c_seed*_oos_predictions.csv" --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/seed_run_summary_a2c.csv
```

### Hyperparameter & Architecture Robustness
```powershell
# Small network
python -m econ499.models.ppo --hparam_file cfg/ppo_small.yaml
# The script will automatically copy the best model to a unique file, e.g.:
#   results/ppo_best_model/ppo_best_model_cfg_ppo_small.zip
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_cfg_ppo_small.zip --hparam_file cfg/ppo_small.yaml --out results/ppo_smallnet_oos_predictions.csv --suffix smallnet
python -m econ499.models.a2c --hparam_file cfg/a2c_small.yaml
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_cfg_a2c_small.zip --hparam_file cfg/a2c_small.yaml --out results/a2c_smallnet_oos_predictions.csv --suffix smallnet

# High learning rate
python -m econ499.models.ppo --hparam_file cfg/ppo_lr_high.yaml
# The script will automatically copy the best model to a unique file, e.g.:
#   results/ppo_best_model/ppo_best_model_cfg_ppo_lr_high.zip
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_cfg_ppo_lr_high.zip --hparam_file cfg/ppo_lr_high.yaml --out results/ppo_lrhigh_oos_predictions.csv --suffix lrhigh
python -m econ499.models.a2c --hparam_file cfg/a2c_lr_high.yaml
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_cfg_a2c_lr_high.zip --hparam_file cfg/a2c_lr_high.yaml --out results/a2c_lrhigh_oos_predictions.csv --suffix lrhigh
```

### Residual Diagnostics
```powershell
python -m econ499.eval.residual_diagnostics --lags 5 --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_residual_lb_lags5.csv
```

### Subsample Analysis
```powershell
python -m econ499.eval.eval_subsamples --years 2010 2015 2020 --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_metrics_subsamples_2010_2015_2020.csv
```

> **Note:**
> The PPO training script now automatically copies the best model to a unique, descriptive filename after each run, based on your experiment parameters (such as --exclude_block, --arb_lambda, --hparam_file, --seed). You no longer need to manually rename or copy the model file after training. The script will print the path to the copied file. Use this path for downstream forecasting and evaluation steps.

## Directory Structure
```
econ499/
├── models/           # DRL agent implementations
├── baselines/        # Baseline models
├── feats/            # Feature engineering scripts
├── eval/             # Evaluation and metrics
├── forecast/         # Forecast generation
├── hpo/              # Hyperparameter optimization
├── utils/            # Helper functions
├── data/             # Data processing scripts
├── results/          # All results, metrics, and figures
├── cfg/              # Configuration files
├── manuscript/       # Paper and figures
└── dev/              # Experimental/sandbox code
```

## Results
The framework generates several outputs:
- Forecast metrics (RMSE, MAE, MASE, MAPE, QLIKE)
- Diebold-Mariano test results
- Model Confidence Set analysis
- Diagnostic plots
- Ablation study results

All results are saved in the `results/` directory.

## Interpretability
To understand which features drive the agent's decisions we expose a small
utility script based on [SHAP](https://github.com/shap/shap).  The script loads
a saved PPO/A2C model, computes SHAP values for a sample of validation
observations, and writes a bar chart of mean absolute attributions.

Run

```bash
python analysis/shap_drl.py --model results/ppo_best_model/best_model.zip
```

Figures are saved in `artifacts/figures`.

## Citation
If you use this code in your research, please cite our paper:
```
@article{your_paper_2024,
  title={Forecasting S\&P 500 Implied Volatility with Deep Reinforcement Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
