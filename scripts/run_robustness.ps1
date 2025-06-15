Write-Host "[ROBUST] Feature block ablation - macro"
python -m econ499.models.ppo --exclude_block macro
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_no_macro.zip --exclude_block macro --out results/ppo_no_macro_oos_predictions.csv --suffix no_macro
python -m econ499.models.a2c --exclude_block macro
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model.zip --exclude_block macro --out results/a2c_no_macro_oos_predictions.csv --suffix no_macro

Write-Host "[ROBUST] Static-arbitrage penalty sensitivity"
python -m econ499.models.ppo --arb_lambda 0
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_arb0.zip --arb_lambda 0 --out results/ppo_arb0_oos_predictions.csv --suffix arb0
python -m econ499.models.ppo --arb_lambda 20
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_arb20.zip --arb_lambda 20 --out results/ppo_arb20_oos_predictions.csv --suffix arb20
python -m econ499.models.a2c --arb_lambda 0
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_l0.zip --arb_lambda 0 --out results/a2c_arb0_oos_predictions.csv --suffix arb0
python -m econ499.models.a2c --arb_lambda 20
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_l20.zip --arb_lambda 20 --out results/a2c_arb20_oos_predictions.csv --suffix arb20

Write-Host "[ROBUST] Alternative sample splits"
python -m econ499.eval.eval_walk_forward --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_metrics_walk.csv
python -m econ499.eval.eval_alt_splits --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_metrics_alt_splits.csv

Write-Host "[ROBUST] Multi-seed evaluation"
python -m econ499.eval.eval_multi_seed --pattern "ppo_seed*_oos_predictions.csv" --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/seed_run_summary_ppo.csv
python -m econ499.eval.eval_multi_seed --pattern "a2c_seed*_oos_predictions.csv" --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/seed_run_summary_a2c.csv

Write-Host "[ROBUST] Hyperparameter and architecture tests"
python -m econ499.models.ppo --hparam_file cfg/ppo_small.yaml
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_cfg_ppo_small.zip --hparam_file cfg/ppo_small.yaml --out results/ppo_smallnet_oos_predictions.csv --suffix smallnet
python -m econ499.models.ppo --hparam_file cfg/ppo_lr_high.yaml
python -m econ499.forecast.make_drl_forecast --model results/ppo_best_model/ppo_best_model_cfg_ppo_lr_high.zip --hparam_file cfg/ppo_lr_high.yaml --out results/ppo_lrhigh_oos_predictions.csv --suffix lrhigh
python -m econ499.models.a2c --hparam_file cfg/a2c_small.yaml
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_cfg_a2c_small.zip --hparam_file cfg/a2c_small.yaml --out results/a2c_smallnet_oos_predictions.csv --suffix smallnet
python -m econ499.models.a2c --hparam_file cfg/a2c_lr_high.yaml
python -m econ499.forecast.make_drl_forecast --model results/a2c_best_model_cfg_a2c_lr_high.zip --hparam_file cfg/a2c_lr_high.yaml --out results/a2c_lrhigh_oos_predictions.csv --suffix lrhigh

Write-Host "[ROBUST] Residual diagnostics"
python -m econ499.eval.residual_diagnostics --lags 5 --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_residual_lb_lags5.csv

Write-Host "[ROBUST] Subsample analysis"
python -m econ499.eval.eval_subsamples --years 2010 2015 2020 --panel_csv results/spx_iv_drl_state.csv --out artifacts/tables/forecast_metrics_subsamples_2010_2015_2020.csv
Write-Host "[ROBUST] Done"

