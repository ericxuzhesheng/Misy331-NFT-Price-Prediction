"""Portable figures; seed spread is not a confidence interval."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_figures(frame, splits, comparison, predictions, history, selected, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ['#2864A1' if 'MLP' not in name else '#B15D29'
              for name in comparison.model]
    ax.barh(comparison.model, comparison.rmsle_mean, color=colors,
            xerr=comparison.rmsle_seed_sd, capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel='Test RMSLE (lower is better)', title='ML and neural networks: later-sale holdout')
    ax.grid(axis='x', alpha=.2)
    fig.text(.01, .005, 'Bars: mean across runs. Whiskers: seed SD, not sampling confidence intervals.',
             fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(output/'model_comparison.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(np.log1p(frame.price_eth), bins=40, color='#2864A1')
    axes[0].set(xlabel='log(1 + sale amount in ETH)', ylabel='Eligible unique NFTs',
                title='Target distribution')
    for name, part in splits.items():
        counts = part.groupby(part.sale_time.dt.floor('D')).size()
        axes[1].plot(counts.index, counts.values, label=name)
    axes[1].set(ylabel='Snapshot last-sale records per day', title='Chronological partitions')
    axes[1].legend(); axes[1].tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(output/'data_and_split.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, name in zip(axes, ['Shallow MLP', 'Deep MLP']):
        candidates = history[history.model == name]
        candidate = selected[name]['candidate']
        curve = candidates[candidates.candidate == candidate]
        ax.plot(curve.epoch, curve.train_log_mse, label='training')
        ax.plot(curve.epoch, curve.validation_log_mse, label='earlier validation')
        ax.axvline(selected[name]['epochs'], color='gray', linestyle=':', label='selected epoch')
        ax.set(xlabel='Epoch', ylabel='MSE on log(1 + ETH)', title=name)
        ax.legend()
    fig.tight_layout(); fig.savefig(output/'mlp_learning_curves.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ['Gradient boosting', 'Deep MLP']):
        actual, fitted = np.log1p(predictions.price_eth), np.log1p(predictions[name])
        ax.scatter(actual, fitted, s=10, alpha=.35, color='#2864A1')
        bound = max(actual.max(), fitted.max())
        ax.plot([0,bound],[0,bound], color='gray', linestyle='--')
        ax.set(xlabel='Observed log(1 + ETH)', ylabel='Predicted log(1 + ETH)',
               title=name + ' (seed 42)', xlim=(0,bound), ylim=(0,bound))
    fig.tight_layout(); fig.savefig(output/'prediction_diagnostics.png'); plt.close(fig)
