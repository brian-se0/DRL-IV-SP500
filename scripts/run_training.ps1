Write-Host "[TRAIN] Training all models"
python -m econ499.baselines.lstm
python -m econ499.baselines.ffn
python -m econ499.models.ppo --timesteps 100000
python -m econ499.models.a2c --timesteps 100000
Write-Host "[TRAIN] Done"
