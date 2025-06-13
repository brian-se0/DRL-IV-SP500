import logging
from pathlib import Path

import numpy as np
import pandas as pd

from iv_drl.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config('data_config.yaml')


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df


def merge():
    data_dir = Path(CONFIG['paths']['output_dir']).resolve()
    output_path = data_dir / 'spx_iv_drl_state.csv'

    paths = {
        'SPX': data_dir / 'spx_daily_features.parquet',
        'IV': data_dir / 'iv_surface_daily_features.parquet',
        'Macro': data_dir / 'macro_daily_features.parquet',
        'FPCA': data_dir / 'iv_fpca_factors.parquet',
    }

    loaded = {k: pd.read_parquet(p) for k, p in paths.items() if p.exists()}
    if not loaded:
        logging.error('No input files found; abort merge.')
        return

    all_dates = pd.Index([])
    for df in loaded.values():
        all_dates = all_dates.union(df.index)

    final = pd.DataFrame(index=all_dates.sort_values())
    for name, df in loaded.items():
        final = final.join(df, how='outer')

    final.index = pd.to_datetime(final.index)
    final = add_calendar(final)
    final.ffill(inplace=True)

    # vol-of-vol splice
    if 'VIXCLS' in final.columns:
        vix_ret = np.log(final['VIXCLS']).diff()
        final['rvvix'] = np.sqrt(252/22) * np.sqrt((vix_ret ** 2).rolling(22).sum()) * 100
    else:
        final['rvvix'] = np.nan

    if 'VVIX' in final.columns:
        final['vol_of_vol'] = np.where(final['VVIX'].notna(), final['VVIX'], final['rvvix'])
        final.drop(columns=['VVIX'], inplace=True)
    else:
        final['vol_of_vol'] = final['rvvix']

    if {'VIXCLS', 'rv_22d'}.issubset(final.columns):
        final['vrp'] = (final['VIXCLS']/100) ** 2 - final['rv_22d']

    if 'VIX3M' not in final.columns and 'atm_iv_90d' in final.columns:
        final['VIX3M'] = final['atm_iv_90d'] * 100
    if 'vix_ts_slope' not in final.columns and {'VIXCLS', 'VIX3M'}.issubset(final.columns):
        final['vix_ts_slope'] = final['VIX3M'] - final['VIXCLS']
    if 'VIX_MOVE_ratio' not in final.columns and {'VIXCLS', 'MOVE'}.issubset(final.columns):
        final['VIX_MOVE_ratio'] = final['VIXCLS'] / final['MOVE']

    mandatory = CONFIG['features']['all_feature_cols'] + [CONFIG['features']['target_col']]
    mandatory_present = [c for c in mandatory if c in final.columns]
    final.dropna(subset=mandatory_present, inplace=True)

    final.reset_index(inplace=True)
    # ensure date column is named consistently for downstream scripts
    if 'index' in final.columns and 'date' not in final.columns:
        final.rename(columns={'index': 'date'}, inplace=True)
    final.to_csv(output_path, index=False, date_format='%Y-%m-%d')
    logging.info('Merged panel saved -> %s', output_path)


def main():
    merge()


if __name__ == '__main__':
    main() 