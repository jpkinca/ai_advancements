import json
import pandas as pd
from datetime import datetime
from sqlalchemy import text
import db_utils as du


def test_feature_signal_target_persistence(engine, sample_prices_df):
    du.ensure_tables(engine)
    # Prepare features
    feat_df = sample_prices_df.copy()
    feat_df['return_1'] = 0.01
    inserted_features = du.upsert_features(engine, feat_df, ['return_1'])
    assert inserted_features == len(feat_df)

    # Upsert again should update but keep count same
    inserted_features_2 = du.upsert_features(engine, feat_df, ['return_1'])
    assert inserted_features_2 == len(feat_df)

    # Signal scores
    ts0 = feat_df['datetime'].iloc[0]
    scores = [
        {
            'instrument': 'AAA',
            'timestamp': ts0,
            'signal_name': 'mom',
            'score': 0.7,
            'meta_json': json.dumps({'lookback': 5})
        },
        {
            'instrument': 'BBB',
            'timestamp': ts0,
            'signal_name': 'mom',
            'score': -0.2,
            'meta_json': None
        }
    ]
    n_scores = du.upsert_signal_scores(engine, scores)
    assert n_scores == 2

    # Upsert update path
    scores[0]['score'] = 0.8
    du.upsert_signal_scores(engine, scores[:1])

    # Episode + target weights + equity
    ep_id = du.create_episode(engine, start_time=datetime.utcnow())
    weights = {'AAA': 0.6, 'BBB': 0.4}
    inserted_w = du.insert_target_weights(engine, ep_id, datetime.utcnow(), weights, strategy='test_strat', rationale='unit test')
    assert inserted_w == 2

    du.insert_equity_point(engine, ep_id, datetime.utcnow(), 10000.0)

    du.finalize_episode(engine, ep_id, end_time=datetime.utcnow())

    # Basic selects to ensure rows exist
    with engine.begin() as conn:
        cnt_feat = conn.execute(text("SELECT COUNT(*) FROM tt_features")).scalar()
        cnt_sig = conn.execute(text("SELECT COUNT(*) FROM tt_signal_scores")).scalar()
        cnt_tw = conn.execute(text("SELECT COUNT(*) FROM tt_target_weights WHERE episode_id=:ep"), {'ep': ep_id}).scalar()
        cnt_eq = conn.execute(text("SELECT COUNT(*) FROM tt_equity_curve WHERE episode_id=:ep"), {'ep': ep_id}).scalar()
    assert cnt_feat >= len(feat_df)
    assert cnt_sig >= 2
    assert cnt_tw == 2
    assert cnt_eq == 1
