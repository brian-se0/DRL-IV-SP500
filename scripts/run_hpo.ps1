Write-Host "[HPO] Starting hyper-parameter optimisation"
python -m econ499.hpo.hpo_lstm --trials 30
python -m econ499.hpo.hpo_ppo --n-trials 30
python -m econ499.hpo.hpo_a2c --n-trials 30
python -m econ499.hpo.hpo_ffn --trials 30
Write-Host "[HPO] Done"
