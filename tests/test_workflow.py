import numpy as np
import pandas as pd
from src.data import FEATURES, NUMERIC, CATEGORICAL, load_data, chronological_split, preprocessor
from src.models import fit_network, predict_network
from src.train import inverse_target, metrics


def sample_data():
    count = 60
    raw = pd.DataFrame({c: np.arange(count) % 7 for c in NUMERIC})
    for c in CATEGORICAL:
        raw[c] = 'common'
    raw['token_id'] = np.arange(count)
    raw['last_sale.total_price'] = 1e18
    raw['last_sale.transaction.timestamp'] = pd.date_range('2022-01-01', periods=count).astype(str)
    raw['owner.address'] = 'after_sale_owner'
    raw['num_sales'] = 999
    return raw


def test_sale_units_filtering_and_duplicate_token(tmp_path):
    raw = sample_data()
    duplicate = raw.iloc[[0]].copy()
    duplicate['last_sale.total_price'] = 2e18
    duplicate['last_sale.transaction.timestamp'] = '2022-04-01'
    raw.loc[1, 'last_sale.total_price'] = np.nan
    raw.loc[2, 'last_sale.total_price'] = 0
    raw = pd.concat([raw, duplicate], ignore_index=True)
    path = tmp_path/'input.csv'; raw.to_csv(path, index=False)
    frame, audit = load_data(path)
    assert len(frame) == 58
    assert frame.loc[frame.token_id == 0, 'price_eth'].iloc[0] == 2.0
    assert audit['duplicate_priced_tokens_removed'] == 1
    assert audit['missing_sale_price_rows'] == 1
    assert not {'owner.address', 'num_sales', 'token_metadata'} & set(FEATURES)


def test_chronological_partition_no_shared_day_or_asset(tmp_path):
    raw = sample_data()
    # Two assets transacted on the same day at a likely partition boundary.
    raw.loc[37, 'last_sale.transaction.timestamp'] = raw.loc[36, 'last_sale.transaction.timestamp']
    path = tmp_path/'input.csv';raw.to_csv(path, index=False)
    frame, _ = load_data(path)
    split = chronological_split(frame)
    assert sum(len(s) for s in split.values()) == len(frame)
    for left, right in [('train','validation'), ('validation','test')]:
        assert split[left].sale_time.max() < split[right].sale_time.min()
        assert set(split[left].sale_time.dt.date).isdisjoint(split[right].sale_time.dt.date)
        assert set(split[left].token_id).isdisjoint(split[right].token_id)


def test_preprocessing_unknown_category_and_training_only_scaling():
    train = sample_data().iloc[:10].copy()
    future = train.iloc[:1].copy()
    future[NUMERIC] = 10000
    future[CATEGORICAL[0]] = 'only_in_future'
    prep = preprocessor()
    a = prep.fit_transform(train[FEATURES])
    b = prep.transform(future[FEATURES])
    assert a.shape[1] == b.shape[1] and np.isfinite(b).all()
    assert 'only_in_future' not in ' '.join(prep.get_feature_names_out())
    np.testing.assert_allclose(prep.named_transformers_['numeric']['scale'].mean_,
                               train[NUMERIC].mean())


def test_metric_units_and_nonnegative_inverse():
    np.testing.assert_allclose(inverse_target(np.log1p([1.,2.])), [1.,2.])
    assert inverse_target([-1.])[0] == 0
    result = metrics(np.array([1.,3.]), np.array([1.,1.]))
    assert np.isclose(result['rmse_eth'], np.sqrt(2.))
    assert result['mae_eth'] == 1.


def test_network_reproducibility_and_finite_predictions():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32,4)).astype('float32')
    y = np.log1p(np.abs(x[:,0]) + .1)
    params = {'hidden':[8,4], 'dropout':.1}
    first = fit_network(x, y, params, 42, epochs=4)
    second = fit_network(x, y, params, 42, epochs=4)
    pred = predict_network(first, x)
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(pred, predict_network(second, x))
