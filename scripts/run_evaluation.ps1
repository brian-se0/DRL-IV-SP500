# Run evaluation and save results
Write-Host "[EVAL] Running evaluator"

Write-Host "Running evaluation with HAR-RV as baseline..."
python econ499/eval/evaluate_all.py --dm_base har_rv --mcs --mcs_alpha 0.1

Write-Host "Saving parameter snapshot..."
python scripts/snapshot_best_params.py

Write-Host "Evaluation complete. Metrics table -> results/tables/forecast_metrics.csv"
