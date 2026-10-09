"""Run a fixed, transparent train/validation/test comparison on a sale snapshot."""
import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import FEATURES, load_data, chronological_split, preprocessor
from .models import candidates, estimator, fit_network, predict_network


def inverse_target(log_prediction):
    prediction = np.expm1(np.maximum(np.asarray(log_prediction), 0))
    if not np.isfinite(prediction).all():
        raise ValueError('Model produced nonfinite ETH predictions.')
    return prediction


def metrics(y, prediction):
    return {'rmse_eth': float(np.sqrt(mean_squared_error(y, prediction))),
            'mae_eth': float(mean_absolute_error(y, prediction)),
            'rmsle': float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(prediction)))),
            'r2': float(r2_score(y, prediction))}


def fit_and_predict(name, params, train, evaluation, seed, *, epochs=200, select=False):
    prep = preprocessor()
    x = prep.fit_transform(train[FEATURES])
    other = prep.transform(evaluation[FEATURES])
    log_y = np.log1p(train.price_eth.to_numpy())
    if 'MLP' in name:
        validation = (other, np.log1p(evaluation.price_eth.to_numpy())) if select else None
        fit = fit_network(x, log_y, params, seed, validation=validation, epochs=epochs)
        prediction = inverse_target(predict_network(fit, other))
        return prediction, fit[3], fit[4]
    fit = estimator(name, params, seed).fit(x, log_y)
    return inverse_target(fit.predict(other)), None, []


def run_experiment(data_path='data/witches_traits.csv', output='results', seeds=(42, 7, 2026),
                   max_epochs=200):
    start = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame, audit = load_data(data_path)
    split = chronological_split(frame)
    train, val, test = (split[k] for k in ['train', 'validation', 'test'])
    audit['partitions'] = {
        key: {'rows': len(part), 'start': part.sale_time.min().isoformat(),
              'end': part.sale_time.max().isoformat()}
        for key, part in split.items()
    }
    assert train.sale_time.max() < val.sale_time.min() and val.sale_time.max() < test.sale_time.min()
    selected, selection_rows, histories = {}, [], []
    # The test partition is not used in candidate or early-stopping decisions.
    for name, configurations in candidates().items():
        rows = []
        for number, params in enumerate(configurations):
            prediction, epoch, history = fit_and_predict(
                name, params, train, val, seeds[0], epochs=max_epochs, select=True)
            row = {'model': name, 'candidate': number, 'parameters': json.dumps(params),
                   'selected_epoch': epoch, **metrics(val.price_eth, prediction)}
            selection_rows.append(row)
            rows.append(row)
            if history:
                histories.extend({'model': name, 'candidate': number, **item} for item in history)
        best = min(rows, key=lambda row: row['rmsle'])
        selected[name] = {'candidate': best['candidate'], 'parameters': json.loads(best['parameters']),
                          'epochs': best['selected_epoch'],
                          'validation_rmsle': best['rmsle']}
        print(f"Selected {name}: validation RMSLE={best['rmsle']:.4f}", flush=True)
    full_train = pd.concat([train, val], ignore_index=True)
    scores, predictions = [], test[['token_id', 'sale_time', 'price_eth']].copy()
    for strategy in ['mean', 'median']:
        name = f'{strategy.title()} baseline'
        dummy = DummyRegressor(strategy=strategy).fit(
            np.zeros((len(full_train), 1)), full_train.price_eth)
        pred = dummy.predict(np.zeros((len(test), 1)))
        scores.append({'model': name, 'seed': seeds[0], **metrics(test.price_eth, pred)})
        predictions[name] = pred
    for name, config in selected.items():
        stochastic = name in ['SGD regression', 'Random forest', 'Gradient boosting',
                              'Shallow MLP', 'Deep MLP']
        fit_seeds = seeds if stochastic else [seeds[0]]
        for seed in fit_seeds:
            pred, _, _ = fit_and_predict(name, config['parameters'], full_train, test, seed,
                                        epochs=config['epochs'] or max_epochs)
            scores.append({'model': name, 'seed': seed, **metrics(test.price_eth, pred)})
            if seed == seeds[0]:
                predictions[name] = pred
        print(f'Evaluated {name} on held-out test', flush=True)
    score_frame = pd.DataFrame(scores)
    comparison = score_frame.groupby('model', sort=False).agg(
        runs=('seed', 'size'), rmsle_mean=('rmsle', 'mean'),
        rmsle_seed_sd=('rmsle', lambda s: float(s.std(ddof=0))),
        rmse_eth_mean=('rmse_eth', 'mean'), mae_eth_mean=('mae_eth', 'mean'),
        r2_mean=('r2', 'mean')).reset_index().sort_values('rmsle_mean')
    score_frame.to_csv(out/'test_metrics.csv', index=False)
    comparison.to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(out/'validation_candidates.csv', index=False)
    pd.DataFrame(histories).to_csv(out/'learning_curves.csv', index=False)
    predictions.to_csv(out/'predictions.csv', index=False)
    audit['versions'] = {name: version(name) for name in
                         ['numpy', 'pandas', 'scikit-learn', 'torch', 'matplotlib']}
    audit['python'] = platform.python_version()
    audit['seeds'] = list(seeds)
    audit['selection_seed'] = seeds[0]
    audit['selection_metric'] = 'validation RMSLE'
    audit['training_target'] = 'log1p(price_eth); inverse expm1, floored at zero'
    audit['selected_models'] = selected
    audit['runtime_seconds'] = round(time.perf_counter()-start, 2)
    (out/'experiment.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    from .report import make_figures
    make_figures(frame, split, comparison, predictions, pd.DataFrame(histories), selected, out)
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/witches_traits.csv')
    parser.add_argument('--output', default='results')
    parser.add_argument('--max-epochs', type=int, default=200)
    args = parser.parse_args()
    print(run_experiment(args.data, args.output, max_epochs=args.max_epochs).to_string(index=False))
