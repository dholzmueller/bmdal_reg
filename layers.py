import torch.distributions
import numpy as np
from .bmdal.layer_features import *


class SplittableModule(nn.Module):
    """
    Base class for vectorized (multiple NNs in parallel) training modules that can be split into non-vectorized modules
    """
    def get_single_model(self, i: int) -> nn.Module:
        raise NotImplementedError()


def get_act_layer(act='relu', **config):
    if act == 'relu':
        return nn.ReLU()
    if act == 'selu':
        return nn.SELU()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'gelu':
        return nn.GELU()
    if act == 'relu6':
        return nn.ReLU6()
    if act == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=config.get('lrelu_a', 1e-2))
    if act == 'rrelu':
        return nn.RReLU()
    if act == 'elu':
        return nn.ELU()
    if act == 'hardtanh':
        return nn.Hardtanh()
    if act == 'softplus':
        return nn.Softplus()
    if act == 'silu':
        return nn.SiLU()

    raise RuntimeError(f'Unknown activation "{act}"')


def get_parallel_act_layer(act='relu', **config):
    layer = get_act_layer(act, **config)
    return ParallelLayerWrapper(layer)


class ParallelLayerWrapper(SplittableModule):
    """
    Wraps a module without parameters into a SplittableModule.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def get_single_model(self, i: int) -> nn.Module:
        return self


class ParallelLinearLayer(SplittableModule):
    """
    Linear layer, vectorized for training multiple NNs in parallel
    """
    def __init__(self, n_models, in_features, out_features, weight_gain=0.25, weight_init_mode='normal',
                 weight_init_gain=1.0, bias_gain=0.1, bias_init_gain=1.0, bias_init_mode='zero', **config):
        super().__init__()
        self.n_models = n_models
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(n_models, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(n_models, out_features))

        with torch.no_grad():
            if weight_init_mode == 'normal':
                self.weight.normal_()
                self.weight *= weight_init_gain
            elif weight_init_mode == 'zero':
                self.weight.zero_()
            else:
                raise RuntimeError(f'Unknown weight_init_mode {weight_init_mode}')

            if bias_init_mode == 'normal':
                self.bias.normal_()
                self.bias *= bias_init_gain
            elif bias_init_mode == 'zero':
                self.bias.zero_()
            else:
                raise RuntimeError(f'Unknown weight_init_mode {weight_init_mode}')

        self.weight_factor = weight_gain/np.sqrt(in_features)
        self.bias_factor = bias_gain

    def get_single_model(self, i: int):
        return LinearLayer(weight=self.weight[i], bias=self.bias[i],
                           weight_factor=self.weight_factor, bias_factor=self.bias_factor)

    def forward(self, x):
        result = self.weight_factor * x.matmul(self.weight) + self.bias_factor * self.bias[:, None, :]
        return result


class LambdaLayer(SplittableModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def get_single_model(self, i: int) -> nn.Module:
        return self

    def forward(self, x):
        return self.f(x)


class ParallelSequential(SplittableModule):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x

    def get_single_model(self, i: int) -> nn.Module:
        return ParallelSequential(*[l.get_single_model(i) for l in self.layers])
