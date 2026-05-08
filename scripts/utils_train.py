import numpy as np
import os
import lib
from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion
from tab_ddpm.graph_denoiser import GraphAwareDenoiser

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes,
    graph_params=None,
    adjacency=None,
):
    print(model_name)
    if graph_params is not None and graph_params.get('enabled', False):
        # Strip keys that belong only to the graph section before passing the
        # remaining model_params to GraphAwareDenoiser.
        mp = {k: v for k, v in model_params.items() if k not in ('rtdl_params',)}
        effective_cat_sizes = [k for k in category_sizes if k > 0]
        model = GraphAwareDenoiser(
            d_in=mp['d_in'],
            num_classes=mp['num_classes'],
            is_y_cond=mp['is_y_cond'],
            d_num=n_num_features,
            cat_sizes=effective_cat_sizes,
            d_model=graph_params.get('d_model', 128),
            n_layers=graph_params.get('n_layers', 4),
            n_heads=graph_params.get('n_heads', 4),
            graph_mode=graph_params.get('mode', 'static'),
            adjacency=adjacency,
            top_k=graph_params.get('sparsity_top_k', 0),
        )
    elif model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    data_path: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = lib.change_val(D)
    
    return lib.transform_dataset(D, T, None)