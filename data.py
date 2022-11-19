import torch
import math
import numpy as np
from pathlib import Path
from typing import *

from . import custom_paths
from . import utils


def batch_randperm(n_batch: int, n: int, device: str = 'cpu') -> torch.Tensor:
    """
    Returns multiple random permutations.
    :param n_batch: Number of permutations.
    :param n: Length of permutations.
    :param device: PyTorch Device to put the permutations on.
    :return: Returns a torch.Tensor of integer type of shape [n_batch, n] containing the n_batch permutations.
    """
    # batched randperm:
    # https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    # https://github.com/pytorch/pytorch/issues/42502
    return torch.stack([torch.randperm(n, device=device) for i in range(n_batch)], dim=0)


def seeded_randperm(n: int, device: str, seed: int) -> torch.Tensor:
    """
    Returns a random permutation according to a seed
    :param n: Length of the permutation.
    :param device: PyTorch device to put the permutation on
    :param seed: Seed for the random sampling.
    :return: Returns a torch.Tensor of shape [n] containing the permutation.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(n, generator=generator).to(device)


class DictDataset:
    """
    Represents a data set containing multiple tensors that can be accessed by their name (a string),
    for example {'x': inputs, 'y': targets}.
    """
    def __init__(self, tensors: Dict[str, torch.Tensor], device: str = None):
        """
        :param tensors: Dictionary of names and tensors.
        All tensors should have the same shape[0], i.e., the same number of samples.
        All tensors should have two dimensions,
        i.e., scalar targets have to be passed with a second dimension of shape 1.
        :param device: PyTorch device that the tensors should be moved to.
        If device is None, all tensors are moved to the device of the first tensor.
        """
        self.device = device if device is not None else next(iter(tensors.values())).device
        self.n_samples = next(iter(tensors.values())).shape[0]
        self.tensors = None if tensors is None else {key: t.to(device) for key, t in tensors.items()}

    def get_batch(self, idxs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns the tensors corresponding to the batch indexed by idxs.
        :param idxs: Tensor of indices to index the tensors of this object with.
        :return: Returns a dictionary {key: t[idxs, :] for key, t in self.tensors.items()}
        """
        return {key: t[idxs, :] for key, t in self.tensors.items()}

    def get_sub_dataset(self, idxs: torch.Tensor) -> 'DictDataset':
        """
        Returns a data set representing the batch given by the indices idxs.
        :param idxs: Tensor of indices used to index the tensors of this object with.
        :return: Returns the DictDataset with the subset of samples specified by idxs.
        """
        return DictDataset(self.get_batch(idxs), device=self.device)

    def __len__(self) -> int:
        """
        :return: Returns the number of samples of the tensors.
        """
        return self.n_samples

    def to(self, device: str) -> 'DictDataset':
        """
        Move all tensors to the given device.
        :param device: PyTorch Device.
        :return: Returns a DictDataset with all tensors moved to the given device.
        """
        return DictDataset(self.tensors, device=device)


class ParallelDictDataLoader:
    """
    This class enables vectorized data loading from DictDatasets,
    i.e., if multiple models are trained in parallel on the same data set,
    this class selects batches independently for each of the models.
    If vectorization is not needed, it is of course possible to use this class with only one model.
    """
    def __init__(self, ds: DictDataset, idxs: torch.Tensor, batch_size: int, shuffle: bool = False,
                 adjust_bs: bool = True, drop_last: bool = False, output_device: Optional[str] = None):
        """
        :param dataset: A DictDataset from which tensors should be loaded
        :param idxs: Vectorized tensor of indices that specify a subset of ds from which data should be loaded.
        The tensor should have shape [n_parallel, n_samples],
        where n_parallel is the number of batches that should be drawn in a vectorized fashion,
        and n_samples is the number of samples that should be selected from ds.
        For example, if ensembling with 3 models should be used,
        ds represents the train+val+pool+test sets and idxs represents the indices belonging to the train set,
        then n_parallel=3 and n_samples=n_train.
        :param batch_size: default batch size, might be automatically adjusted
        :param shuffle: whether the dataset should be shuffled before each epoch
        :param adjust_bs: whether the batch_size may be lowered
        so that the batches are of more equal size while keeping the number of batches the same
        :param drop_last: whether the last batch should be omitted if it is smaller than the other ones
        :param output_device: The device that the returned data should be on
        (if None, take the device where the data already is)
        """
        self.ds = ds
        self.idxs = idxs.to(ds.device)
        self.n_parallel = idxs.shape[0]
        self.n_samples = idxs.shape[1]
        self.output_device = ds.device if output_device is None else output_device
        self.adjust_bs = adjust_bs
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.specified_batch_size = batch_size
        self.batch_size = min(batch_size, self.n_samples)

        if self.drop_last:
            self.n_batches = math.floor(self.n_samples / self.batch_size)
            if adjust_bs:
                self.batch_size = math.floor(self.n_samples / self.n_batches)
            self.sep_idxs = [self.batch_size * i for i in range(self.n_batches + 1)]
        else:
            self.n_batches = math.ceil(self.n_samples / self.batch_size)
            if adjust_bs:
                self.batch_size = math.ceil(self.n_samples / self.n_batches)
            self.sep_idxs = [self.batch_size * i for i in range(self.n_batches)] + [self.n_samples]

    def get_num_samples(self):
        """
        :return: Returns the number of samples that is sampled from (i.e. idxs.shape[1])
        """
        return self.n_samples

    def get_num_iterated_samples(self):
        """
        :return: Returns the number of samples that are visited in one epoch
        (might be less than get_num_samples() if drop_last=True).
        """
        if self.drop_last:
            return self.n_batches * self.batch_size
        return self.get_num_samples()

    def __len__(self):
        """
        :return: Returns the number of batches per epoch.
        """
        return self.n_batches

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        """
        Allows to iterate over batches of an epoch.
        :return: Returns an iterator that allows to iterate over dictionaries of the form {name: tensor}
        with tensor.shape[0] <= batch_size.
        """
        if self.shuffle:
            perms = batch_randperm(self.n_parallel, self.n_samples, device=self.ds.device)
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                batches = self.ds.get_batch(idxs=self.idxs.gather(1, perms[:, start:stop]))
                yield {key: t.to(self.output_device) for key, t in batches.items()}
        else:
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                batches = self.ds.get_batch(idxs=self.idxs[:, start:stop])
                yield {key: t.to(self.output_device) for key, t in batches.items()}


class DataInfo:
    """
    Represents information about a data set.
    """
    def __init__(self, ds_name: str, n_tvp: int, n_test: int, n_features: int,
                 train_test_split: Optional[int]):
        """
        :param ds_name: Name of the data set.
        :param n_tvp: Number of train+val+pool samples.
        :param n_test: Number of test samples.
        :param n_features: Number of input features of the data set.
        :param train_test_split: Set to None if the data set does not have a fixed (train+val+pool)-test split.
        If this is an int, it is interpreted such that the train+val+pool set are the first train_test_split samples
        and the test set are the remaining samples.
        """
        # tvp = train+val+pool
        self.ds_name = ds_name
        self.n_features = n_features
        self.n_tvp = n_tvp
        self.n_test = n_test
        self.n_samples = n_tvp + n_test
        self.train_test_split = train_test_split

    def save(self):
        """
        Saves this object to the path of the corresponding data set.
        """
        utils.serialize(Path(custom_paths.get_data_path()) / 'data' / self.ds_name / 'data_info.pkl', self)


class Task:
    """
    Represents a task, i.e., a data set and information what to do on the data set (how many batch AL steps etc).
    """
    def __init__(self, data_info: DataInfo, task_name: str, n_train: int, n_valid: int,
                 al_batch_sizes: List[int]):
        """
        Constructor. The actual data belonging to the task is loaded lazily, i.e., only when it is needed.
        :param data_info: DataInfo object representing the data set.
        :param task_name: Name of the task.
        This can be similar to the name of the data set
        but may include information about the number and size of batches.
        :param n_train: Number of initial training samples.
        :param n_valid: Number of validation samples.
        The remaining data_info.n_tvp - n_train - n_valid samples are used as initial pool samples.
        :param al_batch_sizes: List of batch sizes to acquire during batch active learning.
        """
        self.task_name = task_name
        self.data_info = data_info
        self.data = None  # load lazily
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_pool = data_info.n_tvp - n_train - n_valid
        self.n_test = data_info.n_test
        self.al_batch_sizes = al_batch_sizes

    def get_data(self) -> DictDataset:
        """
        :return: Returns a DictDataset containing the data set, with names 'X' for the inputs and 'y' for the targets.
        The tensor shape belonging to 'X' is [n_samples, n_features]
        and the tensor shape belonging to 'y' is [n_samples, 1].
        """
        if self.data is None:
            base_path = Path(custom_paths.get_data_path()) / 'data' / self.data_info.ds_name
            X = np.load(f'{base_path}/X.npy')
            y = np.load(f'{base_path}/y.npy')
            self.data = DictDataset({'X': torch.as_tensor(X), 'y': torch.as_tensor(y)})
        return self.data

    @staticmethod
    def get_tabular_tasks(al_batch_sizes: List[int], n_train: int = 1024, n_valid: int = 1024,
                          ds_names: Optional[List[str]] = None, desc: Optional[str] = None) -> List['Task']:
        """
        :param al_batch_sizes: List of batch sizes specifying how many samples should be acquired
        in each batch active learning step.
        len(al_batch_sizes) is the number of batch active learning steps.
        :param n_train: Number of initial training samples.
        :param n_valid: Number of validation samples.
        :param ds_names: Names of data sets that should be used.
        If None is specified, all downloaded data sets are used.
        :param desc: Suffix that should be appended to the dataset names to obtain the task name.
        For example, if desc='256x16' and the current ds_name is 'sgemm',
        then the resulting task name is 'sgemm_256x16'. If desc is None, only the ds_name is used as the task name,
        without '_' at the end.
        :return: Returns a list of tasks, one task per data set.
        """
        base_path = Path(custom_paths.get_data_path())
        data_path = base_path / 'data'
        tasks = []
        if ds_names is None:
            ds_names = [dir.name for dir in data_path.iterdir()]
            ds_names.sort()
        for ds_name in ds_names:
            data_info = utils.deserialize(data_path / ds_name / 'data_info.pkl')
            tasks.append(Task(data_info, ds_name + ('' if desc is None else '_' + desc), n_train, n_valid,
                              al_batch_sizes))
        return tasks


class TaskSplit:
    """
    Represents one particular train-val-pool-test split of a task.
    It also preprocesses the task data according to the split.
    """
    def __init__(self, task: Task, id: int, use_pool_for_normalization: bool = False):
        """
        Creates the split and preprocesses the data. The preprocessed data set is stored in self.data.
        The idxs for train, val, pool, test can be found in
        self.train_idxs, self.valid_idxs, self.pool_idxs, self.test_idxs.
        :param task: Task to split.
        :param id: Identifier of the split. Also serves as a seed for creating the split.
        :param use_pool_for_normalization: Whether to compute the statistics for centering and standardization
        only on the (initial) train set or on train+pool sets. If the train set is small,
        """
        self.al_batch_sizes = task.al_batch_sizes
        self.data = task.get_data()
        self.id = id
        self.task_name = task.task_name
        self.n_samples = task.data_info.n_samples
        self.use_pool_for_normalization = use_pool_for_normalization

        old_state = np.random.get_state()
        np.random.seed(id)
        if task.data_info.train_test_split is not None:
            n_tvp = task.data_info.train_test_split
            tvp_perm = np.random.default_rng(id).permutation(n_tvp)
            self.test_idxs = np.arange(n_tvp, self.n_samples, dtype=np.int)
        else:
            perm = np.random.permutation(self.n_samples)
            n_tvp = self.n_samples - task.n_test
            tvp_perm = perm[:n_tvp]
            self.test_idxs = perm[n_tvp:]
        np.random.set_state(old_state)
        s1 = task.n_train
        s2 = s1 + task.n_valid
        self.train_idxs = tvp_perm[:s1]
        self.valid_idxs = tvp_perm[s1:s2]
        self.pool_idxs = tvp_perm[s2:]

        # preprocess tensors
        X = self.data.tensors['X']
        if use_pool_for_normalization:
            norm_idxs = np.concatenate([self.train_idxs, self.pool_idxs], axis=0)
        else:
            norm_idxs = self.train_idxs
        X_norm = X[norm_idxs]
        X = (X - X_norm.mean(dim=0, keepdim=True)) / (X_norm.std(dim=0, keepdim=True) + 1e-30)
        X = 5 * torch.tanh(0.2 * X)
        self.data = DictDataset({'X': X, 'y': self.data.tensors['y']})

    def get_data(self) -> DictDataset:
        """
        :return: Returns the DictDataset representing all of the data (train+val+pool+test)
        """
        return self.data

    def get_train_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the (initial) training data.
        """
        return self.train_idxs

    def get_valid_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the validation data.
        """
        return self.valid_idxs

    def get_pool_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the (initial) pool data.
        """
        return self.pool_idxs

    def get_test_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the test data.
        """
        return self.test_idxs
