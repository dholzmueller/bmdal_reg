from layers import *


def create_tabular_model(n_models, n_features, hidden_sizes=[512]*2, act='relu', **config):
    layer_sizes = [n_features] + hidden_sizes + [1]
    layers = []
    for in_features, out_features in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        layers.append(ParallelLinearLayer(n_models, in_features, out_features, **config))
        layers.append(get_parallel_act_layer(act))
    layers.append(ParallelLinearLayer(n_models, layer_sizes[-2], layer_sizes[-1],
                                      weight_init_mode='zero' if config.get('use_llz', False) else 'normal', **config))
    return ParallelSequential(*layers)

