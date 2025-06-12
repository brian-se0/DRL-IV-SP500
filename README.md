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
- **Market Sentiment**: VIX, VVIX, and other volatility indices

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
- **Action Space**: Continuous prediction of next-day ATM IV
- **Reward Function**: Negative MSE with static-arbitrage penalty
- **Constraints**: No-arbitrage conditions enforced through penalty term

### Baselines
1. **Traditional Models**
   - HAR-RV
   - AR(1)
   - GARCH(1,1)

2. **Machine Learning**
   - LSTM
   - Ridge Regression
   - OLS

3. **Simple Benchmarks**
   - Naive (today's IV)
   - Rolling mean

## Implementation

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv312
source venv312/bin/activate  # Linux/Mac
.\venv312\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0+cu121 torchvision torchaudio
```

### Data Processing Pipeline
```bash
# 1. Build price features
python price_feats.py

# 2. Build option surface features
python surface_feats.py

# 3. Fetch macro features
python macro_feats.py

# 4. Merge all features
python merge_feats.py
```

### Model Training
```bash
# Train PPO agent
python -m iv_drl.models.ppo --timesteps 100000

# Train A2C agent
python -m iv_drl.models.a2c --timesteps 100000

# Train baselines
python -m iv_drl.baselines.lstm
python -m iv_drl.baselines.garch
python -m iv_drl.baselines.ols
```

### Hyperparameter Optimization
```bash
# LSTM HPO
python -m iv_drl.hpo.hpo_lstm --n_trials 50

# DRL HPO
python -m iv_drl.hpo.hpo_drl --agent ppo --n_trials 30
```

### Evaluation
```bash
# Run full evaluation
python -m iv_drl.eval.evaluate_all

# Run with statistical tests
python -m iv_drl.eval.evaluate_all --dm_base naive --spa_base naive --mcs
```

## Robustness Checks

### Feature Block Ablations
```bash
# Train without macro features
python -m iv_drl.models.ppo --exclude_block macro

# Generate forecasts
python -m iv_drl.forecast.make_drl_forecast --model ppo_best_model/best_model.zip --exclude_block macro
```

### Static-Arbitrage Penalty Sensitivity
```bash
# Train with different penalty weights
python -m iv_drl.models.ppo --lambda 0  # No penalty
python -m iv_drl.models.ppo --lambda 20  # Strong penalty
```

### Alternative Sample Splits
```bash
# Walk-forward evaluation
python -m iv_drl.eval.eval_walk_forward

# Hold-out evaluation
python -m iv_drl.eval.eval_alt_splits
```

## Directory Structure
```
iv_drl/
├── models/           # DRL agent implementations
├── baselines/        # Baseline models
├── feats/            # Feature engineering scripts
├── eval/             # Evaluation and metrics
├── forecast/         # Forecast generation
├── hpo/              # Hyperparameter optimization
├── orchestrate/      # Pipeline orchestration
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
