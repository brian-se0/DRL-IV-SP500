Write-Host "[FEATURE] Building all features"
python -m econ499.feats.build_price
python -m scripts.process_option_data
python -m econ499.feats.build_iv_surface
python -m econ499.feats.build_iv_fpca
python -m econ499.feats.fetch_macro
python -m econ499.panels.merge_features
Write-Host "[FEATURE] Done"
