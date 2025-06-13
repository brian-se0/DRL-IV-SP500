# Run DRL model training
Write-Host "[TRAIN] Training DRL models"

Write-Host "Training PPO (this may take a while)..."
python econ499/models/ppo.py --timesteps 500000 --arb_lambda 10

Write-Host "Training A2C (this may take a while)..."
python econ499/models/a2c.py --timesteps 500000 --arb_lambda 10

Write-Host "DRL training complete. Models saved in results/models/" 