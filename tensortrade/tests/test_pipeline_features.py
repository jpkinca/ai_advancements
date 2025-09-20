import pandas as pd
from mvp_pipeline import compute_basic_features


def test_compute_basic_features(sample_prices_df):
    df = compute_basic_features(sample_prices_df)
    assert 'return_1' in df.columns
    assert 'vol_10' in df.columns
    assert 'vol_20' in df.columns
    # Ensure no exception and row count preserved
    assert len(df) == len(sample_prices_df)
