"""Classical estimators and an explicit PyTorch feed-forward training loop."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor


def candidates():
    return {
        'Linear regression': [{}],
        'Ridge': [{'alpha': a} for a in [0.1, 1., 10., 100.]],
        'SGD regression': [{'alpha': a} for a in [0.0001, 0.001]],
        'Random forest': [{'max_depth': depth, 'min_samples_leaf': leaf}
                          for depth, leaf in [(8, 2), (16, 2), (None, 5)]],
        'Gradient boosting': [{'max_leaf_nodes': nodes, 'l2_regularization': reg}
                              for nodes, reg in [(7, 1.), (15, 1.), (15, 10.)]],
        'Shallow MLP': [{'hidden': [32], 'dropout': 0.0},
                        {'hidden': [64], 'dropout': 0.1}],
        'Deep MLP': [{'hidden': [64, 32], 'dropout': 0.0},
                     {'hidden': [64, 32], 'dropout': 0.1}],
    }


def estimator(name, params, seed):
    if name == 'Linear regression':
        return LinearRegression()
    if name == 'Ridge':
        return Ridge(**params)
    if name == 'SGD regression':
        return SGDRegressor(loss='squared_error', penalty='l2', max_iter=3000,
                            tol=1e-5, random_state=seed, **params)
    if name == 'Random forest':
        return RandomForestRegressor(n_estimators=160, n_jobs=2,
                                     random_state=seed, **params)
    if name == 'Gradient boosting':
        # External chronological validation selects complexity and iteration count is fixed.
        return HistGradientBoostingRegressor(max_iter=160, learning_rate=.05,
                                            early_stopping=False, random_state=seed, **params)
    raise ValueError(name)


def make_network(input_dim, hidden, dropout):
    layers = []
    previous = input_dim
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def fit_network(x, y, params, seed, *, validation=None, epochs=200, patience=20):
    """Chronological early stopping during selection; fixed epochs during final refit."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    net = make_network(x.shape[1], params['hidden'], params['dropout'])
    # Target scaling is learnt on the training slice only.
    target_mean, target_scale = float(np.mean(y)), max(float(np.std(y)), 1e-6)
    tx = torch.tensor(np.asarray(x, dtype=np.float32))
    ty = torch.tensor((np.asarray(y) - target_mean) / target_scale, dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, waiting, state = float('inf'), 1, 0, None
    history = []
    if validation is not None:
        vx, vy = validation
        vx = torch.tensor(np.asarray(vx, dtype=np.float32))
        vy = torch.tensor((np.asarray(vy) - target_mean) / target_scale, dtype=torch.float32)
    for epoch in range(1, epochs + 1):
        net.train()
        for indices in torch.randperm(len(tx), generator=generator).split(128):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(tx[indices]).flatten(), ty[indices])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_loss = nn.functional.mse_loss(net(tx).flatten(), ty).item() * target_scale**2
            validation_loss = (nn.functional.mse_loss(net(vx).flatten(), vy).item()
                               * target_scale**2 if validation is not None else None)
        history.append({'epoch': epoch, 'train_log_mse': train_loss,
                        'validation_log_mse': validation_loss})
        if validation is not None:
            if validation_loss < best_loss - 1e-6:
                best_loss, best_epoch, waiting = validation_loss, epoch, 0
                state = copy.deepcopy(net.state_dict())
            else:
                waiting += 1
            if waiting >= patience:
                break
    if validation is not None:
        net.load_state_dict(state)
    else:
        best_epoch = epochs
    return net, target_mean, target_scale, best_epoch, history


def predict_network(fit, x):
    net, mean, scale = fit[:3]
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(np.asarray(x, dtype=np.float32))).flatten().numpy()
    return pred * scale + mean
