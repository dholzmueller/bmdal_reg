import numpy as np
import torch
from typing import List

from .data import DictDataset, ParallelDictDataLoader
from .models import create_tabular_model
from .task_execution import get_devices
from .train import fit_model


class NNRegressor:
    """
    Scikit-learn style interface for the NN regression (without active learning) used in this repository.
    """
    def __init__(self, lr: float = 0.15, hidden_sizes: List[int] = None, act: str = 'silu', n_ensemble: int = 1,
                 batch_size: int = 256, n_epochs: int = 256, weight_decay: float = 0.0,
                 weight_gain: float = 0.5, bias_gain: float = 1.0, valid_fraction: float = 0.1, device: str = None,
                 preprocess_data: bool = True, seed: int = 0):
        """
        Constructor with sensible default values (optimized as in the paper).
        :param lr: Learning rate.
        :param hidden_sizes: Sizes of hidden layers. If None, set to [512, 512]
        :param act: Activation function such as 'relu' or 'silu'. For more options, see layers.get_act_layer().
        :param n_ensemble: How many NNs should be used in the ensemble. Defaults to 1.
        :param batch_size: Batch size to use (will be automatically adjusted to a smaller one if needed).
        :param n_epochs: Number of epochs to train maximally.
        :param weight_decay: Weight decay parameter.
        :param weight_gain: Factor for the weight parameters of linear layers.
        :param bias_gain: Factor for the bias parameters of linear layers.
        :param valid_fraction: Which fraction of the training data set should be used for validation.
        :param device: Device to train on. Should be a string that PyTorch accepts, such as 'cpu' or 'cuda:0'.
        If None, the first GPU is used if one is found, otherwise the CPU is used.
        :param preprocess_data: Whether X and y values should be standardized for training and X should be soft-clipped.
        :param seed: Random seed for training.
        """
        self.n_models = n_ensemble
        self.hidden_sizes = hidden_sizes or [512, 512]
        self.lr = lr
        self.act = act
        self.weight_gain = weight_gain
        self.bias_gain = bias_gain
        self.model = None
        self.valid_fraction = valid_fraction
        self.device = device or get_devices()[0]
        self.seed = seed
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.preprocess_data = preprocess_data
        self.means = None
        self.stds = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        if len(y.shape) == 1:
            y = y[:, None]
        n_outputs = y.shape[1]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = create_tabular_model(n_models=self.n_models, n_features=n_features,
                                          hidden_sizes=self.hidden_sizes,
                                          act=self.act, n_outputs=n_outputs,
                                          weight_gain=self.weight_gain, bias_gain=self.bias_gain).to(self.device)

        X = torch.as_tensor(X, dtype=torch.float).to(self.device)
        y = torch.as_tensor(y, dtype=torch.float).to(self.device)

        if self.preprocess_data:
            self.means = X.mean(dim=0, keepdim=True)
            self.stds = X.std(dim=0, keepdim=True)
            self.y_mean = y.mean().item()
            self.y_std = y.std().item()
            X = (X - self.means) / (self.stds + 1e-30)
            X = 5 * torch.tanh(0.2 * X)
            y = (y - self.y_mean) / (self.y_std + 1e-30)

        data = DictDataset({'X': X, 'y': y})
        n_valid = int(self.valid_fraction * X.shape[0])
        perm = np.random.permutation(X.shape[0])
        valid_idxs = torch.as_tensor(perm[:n_valid]).to(self.device)
        train_idxs = torch.as_tensor(perm[n_valid:]).to(self.device)

        fit_model(self.model, data, n_models=self.n_models, train_idxs=train_idxs, valid_idxs=valid_idxs,
                  n_epochs=self.n_epochs, batch_size=self.batch_size, lr=self.lr,
                  weight_decay=self.weight_decay, valid_batch_size=8192)

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float).to(self.device)

        if self.preprocess_data:
            X = (X - self.means) / (self.stds + 1e-30)
            X = 5 * torch.tanh(0.2 * X)

        data = DictDataset({'X': X})
        idxs = torch.arange(X.shape[0], device=self.device)

        dl = ParallelDictDataLoader(data, idxs.expand(self.n_models, -1), batch_size=8192, shuffle=False,
                                         adjust_bs=False, drop_last=False)
        with torch.no_grad():
            self.model.eval()
            y_pred = torch.cat([self.model(batch['X']) for batch in dl], dim=1).mean(dim=0)

        if self.preprocess_data:
            y_pred = y_pred * self.y_std + self.y_mean

        return y_pred.detach().cpu().numpy()




