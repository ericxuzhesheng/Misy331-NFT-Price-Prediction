"""Explicit trait whitelist, target construction and chronological partitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['Wonder', 'Wisdom', 'Will', 'Wit', 'Wiles', 'Woe']
CATEGORICAL = ['Skin Tone', 'Rising Sign', 'Eyebrows', 'Body Shape', 'Moon Sign',
               'Hair Color', 'Necklace', 'Sun Sign', 'Eye Style', 'Eye Color',
               'Mouth', 'Hat', 'Archetype of Power', 'Hair (Front)', 'Top',
               'Hair (Back)', 'Background', 'Face Markings', 'Facewear',
               'Hair Topper', 'Back Item', 'Earrings', 'Forehead Jewelry',
               'Hair (Middle)', 'Mask', 'Outerwear']
FEATURES = NUMERIC + CATEGORICAL


def load_data(path: str | Path):
    raw = pd.read_csv(path)
    required = FEATURES + ['token_id', 'last_sale.total_price',
                           'last_sale.transaction.timestamp']
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    price = pd.to_numeric(raw['last_sale.total_price'], errors='coerce') / 1e18
    dates = pd.to_datetime(raw['last_sale.transaction.timestamp'], errors='coerce', utc=True)
    valid_price = price.notna() & np.isfinite(price) & (price > 0)
    valid = valid_price & dates.notna() & raw['token_id'].notna()
    frame = raw.loc[valid, FEATURES + ['token_id']].copy()
    frame['sale_time'] = dates[valid]
    frame['price_eth'] = price[valid]
    # A snapshot can contain repeated assets. Keep one record per token across all splits.
    frame = frame.sort_values(['sale_time', 'token_id'], kind='stable')
    before_dedup = len(frame)
    frame = frame.drop_duplicates('token_id', keep='last').reset_index(drop=True)
    if frame.empty:
        raise ValueError('No eligible priced records with valid timestamps and token IDs.')
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    # A missing cosmetic trait usually means the item does not have that accessory.
    for col in CATEGORICAL:
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    audit = {
        'raw_rows': len(raw), 'missing_sale_price_rows': int(price.isna().sum()),
        'invalid_or_nonpositive_price_rows': int((price.notna() & ~valid_price).sum()),
        'invalid_timestamp_rows_among_priced': int((valid_price & dates.isna()).sum()),
        'missing_token_rows_among_priced': int((valid_price & raw['token_id'].isna()).sum()),
        'duplicate_priced_tokens_removed': before_dedup - len(frame),
        'eligible_unique_tokens': len(frame), 'features': FEATURES,
        'target': 'last_sale.total_price / 1e18 (ETH-equivalent units)',
        'earliest_sale': frame.sale_time.min().isoformat(),
        'latest_sale': frame.sale_time.max().isoformat(),
        'median_price_eth': float(frame.price_eth.median()),
        'max_price_eth': float(frame.price_eth.max()),
        'feature_missing_fraction': frame[FEATURES].isna().mean().to_dict(),
    }
    return frame, audit


def chronological_split(frame):
    """Approximate 60/20/20 by rows, keeping each UTC calendar day together.

    The observations are irregular transactions, not equally spaced time-series samples.
    The resulting windows can have different calendar durations.
    """
    if len(frame) < 30:
        raise ValueError('At least 30 eligible records are required.')
    dates = frame.sale_time.dt.floor('D')
    train_end = dates.iloc[int(len(frame) * .6)]
    validation_end = dates.iloc[int(len(frame) * .8)]
    if train_end >= validation_end:
        raise ValueError('Insufficient distinct days for chronological partitions.')
    splits = {
        'train': frame[dates < train_end].copy(),
        'validation': frame[(dates >= train_end) & (dates < validation_end)].copy(),
        'test': frame[dates >= validation_end].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError('A chronological partition is empty.')
    return splits


def preprocessor():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='__absent__',
                                 keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', min_frequency=5,
                                 sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer([('numeric', numeric, NUMERIC),
                              ('categorical', categorical, CATEGORICAL)])
