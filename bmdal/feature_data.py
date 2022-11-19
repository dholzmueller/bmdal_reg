from typing import *
import torch

from .. import utils


def torch_cat(tensors: List[torch.Tensor], dim: int):
    """
    Implements torch.cat() but doesn't copy if only one tensor is provided.
    This can make it faster if no copying behavior is needed.
    :param tensors: Tensors to be concatenated.
    :param dim: Dimension in which the tensor should be concatenated.
    :return: The concatendated tensor.
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


class Indexes:
    """
    This class allows to store and manipulate ways of indexing tensors, either via slices or lists of integers.
    This is important for computing subsets of kernel matrices and minibatching of such computations.
    """
    def __init__(self, n_samples: int, idxs: Optional[Union[torch.Tensor, slice, int, 'Indexes']]):
        """
        :param n_samples: Total size of the dimension of tensors that should be indexed.
        This is used to convert negative slices like [:-1] to positive slices like [:n_samples-1].
        Also, it is used to compute how many elements indexing with the Indexes object will be returned.
        :param idxs: Object that a tensor should be indexed with. Can be either
        - a tensor of ints,
        - a non-empty slice without stride, i.e., [1:2:-1] is not allowed because of the -1,
                and [2:1] is not allowed because it is empty.
        - a single index i (which will be treated as the slice [i:i+1],
                 i.e., it will not remove the dimension from the indexed tensor),
        - None, corresponding to [:], i.e., selecting all elements
        - or already an Indexes object (which will then be copied)
        """
        self.n_samples = n_samples
        if idxs is None:
            self.idxs = slice(0, n_samples)
        elif isinstance(idxs, Indexes):
            self.idxs = idxs.idxs
        elif isinstance(idxs, int):
            self.idxs = slice(idxs, idxs+1)
        elif isinstance(idxs, slice):
            if idxs.step is not None and idxs.step != 1:
                raise ValueError(f'Cannot handle slices with step size other than 1')
            start = 0 if idxs.start is None else idxs.start + (0 if idxs.start >= 0 else n_samples)
            stop = n_samples if idxs.stop is None else idxs.stop + (0 if idxs.stop >= 0 else n_samples)
            if stop <= start:
                raise ValueError(f'stop <= start not allowed for slices')
            self.idxs = slice(start, stop)
        elif isinstance(idxs, torch.Tensor):
            self.idxs = idxs
        else:
            raise ValueError(f'Cannot handle index type {type(idxs)}')

        if isinstance(self.idxs, slice):
            self.n_idxs = self.idxs.stop - self.idxs.start
        else:
            self.n_idxs = len(self.idxs)

    def __len__(self):
        """
        :return: Number of elements that remain after indexing with self
        """
        return self.n_idxs

    def compose(self, other: 'Indexes') -> 'Indexes':
        """
        :param other: an Indexes object that should be used to subselect indexes from self
        :return: Returns an Indexes object such that data[self][other] = data[self.compose(other)]
        """
        if isinstance(self.idxs, torch.Tensor):
            new_idxs = self.idxs[other.idxs]
        elif isinstance(self.idxs, slice):
            if isinstance(other.idxs, torch.Tensor):
                new_idxs = other.idxs + self.idxs.start
            elif isinstance(other.idxs, slice):
                new_idxs = slice(self.idxs.start + other.idxs.start, self.idxs.start + other.idxs.stop)
            else:
                raise RuntimeError('other.idxs is neither slice nor torch.Tensor')
        else:
            raise RuntimeError('self.idxs is neither slice nor torch.Tensor')

        return Indexes(len(self), new_idxs)

    def get_idxs(self):
        """
        :return: Returns the slice or LongTensor that this object represents, which can be used to index a torch.Tensor
        """
        return self.idxs

    def split_by_sizes(self, sample_sizes: List[int]) -> Iterable[Tuple[int, 'Indexes']]:
        """
        This method allows to apply the Indexes object to multiple arrays as if they were concatenated.
        This is useful for indexing batched data.
        This method is currently only implemented for Indexes that represent slices
        and not those that represent LongTensors.
        Example:
        full_arr = torch.arange(32)
        idxs = Indexes(16, slice(5, 17))
        batched_arrs = [full_arr[4*i:4*(i+1)] for i in range(8)]
        batch_sizes = [len(arr) for arr in batched_arrs]
        full_arr_indexed = full_arr[idxs.get_idxs()]
        batched_arrs_indexed = torch.concat([batched_arrs[i][sub_idxs.get_idxs()]
                                        for i, sub_idxs in idxs.split_by_sizes(batch_sizes)], dim=0)
        assert full_arr_indexed == batched_arrs_indexed
        :param sample_sizes: Sizes of the multiple arrays that should be indexed
        :return: Returns a generator object that can be used to iterate over sub-Indexes objects
        corresponding to the corresponding data parts.
        """
        if isinstance(self.idxs, slice):
            start = self.idxs.start
            stop = self.idxs.stop
            for i, sz in enumerate(sample_sizes):
                if start < sz and stop > 0:
                    yield i, Indexes(sz, slice(max(start, 0), min(stop, sz)))
                start -= sz
                stop -= sz
        else:
            raise NotImplementedError('indexing splitted data with a list of indexes is currently not supported')

    def is_all_slice(self) -> bool:
        """
        :return: Returns true iff applying this object is equivalent to selecting all data, i.e. [:].
        """
        return isinstance(self.idxs, slice) and self.idxs.start == 0 and self.idxs.stop == self.n_samples


class FeatureData:
    """
    Abstract base class for classes that represent data that serves as input to feature maps.
    """
    def __init__(self, n_samples: int, device: str, dtype):
        """
        :param n_samples: Number of samples represented by this FeatureData object.
        :param device: String representing the torch device that this feature data is located on.
        :param dtype: Torch dtype that the feature data has
        """
        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype

    def get_n_samples(self) -> int:
        """
        :return: Returns the number of samples.
        """
        return self.n_samples

    def get_device(self) -> str:
        """
        :return: Returns the device that the data is on.
        """
        return self.device

    def get_dtype(self) -> Any:
        """
        :return: Returns the (torch) dtype that the feature data has.
        """
        return self.dtype

    def __len__(self) -> int:
        """
        :return: Returns the number of samples.
        """
        return self.get_n_samples()

    def __getitem__(self, idxs: Union[torch.Tensor, slice, int, 'Indexes']):
        """
        :param idxs: Represents the subset of samples that should be returned.
        Note that if idxs is an int, the dimension will not be collapsed.
        In other words, self[i] is equivalent to self[i:i+1].
        :return: Returns the feature data represented by the subset of indexes in idxs.
        """
        idxs = Indexes(self.get_n_samples(), idxs)
        if idxs.is_all_slice():
            return self
        return SubsetFeatureData(self, idxs)

    def simplify(self, idxs: Optional[Union[torch.Tensor, slice, int, Indexes]] = None) -> 'FeatureData':
        """
        Simplifies the FeatureData object recursively, for example by concatenating batches.
        Do not override this method in subclasses. Override simplify_impl_() instead.
        :param idxs: The subset of the data that should be simplified.
        :return: Returns a simplified version of self[idxs].
        """
        idxs = Indexes(self.get_n_samples(), idxs)
        return self.simplify_impl_(idxs)

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        """
        Internal method to implement simplify(). This method should be overridden by subclasses.
        :param idxs: Indexes for the subset of the data that should be simplified.
        :return: Returns a simplified version of self[idxs].
        """
        raise NotImplementedError()

    def simplify_multi_(self, feature_data_list: List['FeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        """
        Internal method that is sometimes required to implement simplify() recursively.
        This method should be overridden by subclasses.
        This method should simplify feature_data[idxs] for feature_data, idxs in zip(feature_data_list, idxs_list)
        and then concatenate the feature_data objects suitably.
        The concatenation needs to be handled by this custom method
        because we want the concatenation of multiple ListFeatureData objects
        to be a ListFeatureData where the corresponding list elements get concatenated.
        :param feature_data_list: Lists of feature data objects to be simplified and concatenated.
        The feature data objects should be of the same subclass of FeatureData as self.
        :param idxs_list: List of indexes to apply to the feature data objects.
        :return: A simplified version of a concatenation of the indexed feature_data_list elements.
        """
        raise NotImplementedError()

    def iterate(self, idxs: Indexes) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        """
        This method allows to iterate over sub-parts of the feature data.
        This is necessary because the FeatureData may consist of a concatenation
        of multiple individual tensor feature data objects.
        Example:
        tfd_list = [TensorFeatureData(torch.zeros(5, 3)) for i in range(3)]
        feature_data = ConcatFeatureData(tfd_list)
        idxs = Indexes(len(feature_data), None)
        tensor = torch.concat([tfd.get_tensor(sub_idxs) for sub_idxs, tfd in feature_data.iterate(idxs)], dim=0)
        # in this case, one could have just used tensor = feature_data.get_tensor(idxs) directly
        # however, this is useful for computing kernel matrices in batches, for example.
        :param idxs: Sub-indexes of the data that should be iterated over.
        :return: Returns a generator object that allows to iterate over tuples of (Indexes, FeatureData).
        """
        yield idxs, self  # default implementation

    def __iter__(self) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        """
        This method corresponds to iterate(Indexes(len(self), None)), i.e., iterating over the whole data.
        To use this method, just use something like "for sub_idxs, sub_data in feature_data", without the .__iter__().
        :return: Returns a generator object that allows to iterate over parts of the data, see iterate().
        """
        return self.iterate(Indexes(n_samples=self.get_n_samples(), idxs=None))

    def batched(self, batch_size: int) -> 'FeatureData':
        """
        Batch this feature data, such that __iter__() or iterate() iterates over this data in multiple batches
        :param batch_size: Maximum batch size (the last batch may be smaller).
        :return: A BatchedFeatureData object.
        """
        return BatchedFeatureData(self, batch_size=batch_size)

    def get_tensor(self, idxs: Optional[Union[torch.Tensor, slice, int, Indexes]] = None) -> torch.Tensor:
        """
        Returns the tensor corresponding to this feature data indexed by idxs.
        This method is only well-defined for feature data that does not involve ListFeatureData,
        since otherwise it would need to return a list of tensors.
        This method should not be overridden by subclasses, they should override get_tensor_impl_() instead.
        :param idxs: object to index the tensor with, or None if the full tensor should be returned.
        :return: Returns a torch.Tensor representing the indexed data.
        The first dimension of the tensor is the n_samples dimension.
        """
        idxs = Indexes(self.get_n_samples(), idxs)
        return self.get_tensor_impl_(idxs)

    def get_tensor_impl_(self, idxs: Indexes) -> torch.Tensor:
        """
        Internal method for computing the result of get_tensor().
        Should be overridden by classes
        that do not want to use the default behavior that is implemented here via self.iterate(idxs)
        :param idxs: Indexes object to index the tensor with.
        :return: The resulting tensor.
        """
        return torch_cat([feature_data.get_tensor(sub_idxs) for sub_idxs, feature_data in self.iterate(idxs)], dim=0)

    def cast_to(self, dtype) -> 'FeatureData':
        """
        Casts the data to another dtype.
        :param dtype: Target type, should be usable by torch.
        :return: A FeatureData object with internal data cast to the target dtype.
        """
        raise NotImplementedError()

    def to_indexes(self, idxs: Optional[Indexes]) -> Indexes:
        """
        :param idxs: Indexes object or None
        :return: the Indexes object or an Indexes object representing None
        """
        return idxs if idxs is not None else Indexes(self.get_n_samples(), None)


class EmptyFeatureData(FeatureData):
    """
    This object represents feature data with no samples.
    This is used internally during simplification and should not matter to users.
    Some methods like get_tensor() may not work for this class.
    """
    def __init__(self, device: str, dtype):
        super().__init__(n_samples=0, device=device, dtype=dtype)

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return self

    def simplify_multi_(self, feature_data_list: List['EmptyFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        return EmptyFeatureData(device=self.device, dtype=self.dtype)

    def cast_to(self, dtype) -> 'FeatureData':
        return self


class TensorFeatureData(FeatureData):
    """
    FeatureData subclass representing data consisting of a single tensor of shape [n_samples, n_features].
    """
    def __init__(self, data: torch.Tensor):
        """
        :param data: Tensor of shape [n_samples, ...], usually [n_samples, n_features]
        """
        super().__init__(n_samples=data.shape[-2], device=data.device, dtype=data.dtype)
        self.data = data

    def get_tensor_impl_(self, idxs: Indexes) -> torch.Tensor:
        return self.data[idxs.get_idxs()]

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return TensorFeatureData(self.get_tensor_impl_(idxs))

    def simplify_multi_(self, feature_data_list: List['TensorFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        return TensorFeatureData(torch_cat([fd.data[idxs.get_idxs()]
                                            for fd, idxs in zip(feature_data_list, idxs_list)], dim=0))

    def cast_to(self, dtype) -> 'FeatureData':
        return TensorFeatureData(self.data.type(dtype))


class SubsetFeatureData(FeatureData):
    """
    FeatureData subclass representing a subset of other FeatureData,
    where the subset is given by an Indexes object.
    """
    def __init__(self, feature_data: FeatureData, idxs: Indexes):
        """
        :param feature_data: FeatureData which this object represents a subset of
        :param idxs: Indexes object representing the subset of indexes.
        """
        super().__init__(n_samples=len(idxs), device=feature_data.get_device(), dtype=feature_data.get_dtype())
        self.idxs = idxs
        self.feature_data = feature_data

    def iterate(self, idxs: Optional[Indexes] = None) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        for sub_idxs, feature_data in self.feature_data.iterate(self.idxs.compose(idxs)):
            yield sub_idxs, feature_data

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return self.feature_data.simplify(self.idxs.compose(idxs))

    def simplify_multi_(self, feature_data_list: List['SubsetFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        return ConcatFeatureData([fd.simplify(idxs) for fd, idxs in zip(feature_data_list, idxs_list)]).simplify()

    def cast_to(self, dtype) -> 'FeatureData':
        return SubsetFeatureData(self.feature_data.cast_to(dtype), self.idxs)


class ConcatFeatureData(FeatureData):
    """
    FeatureData subclass that allows to concatenate multiple FeatureData objects into one object.
    Note that this does not concatenate the tensors,
    and .iterate() / .__iter__() may therefore loop over multiple feature data objects.
    While this class can be used to force batched processing of data,
    it is recommended to use BatchedFeatureData for this purpose since it does not actually split up the tensors.
    """
    def __init__(self, feature_data_list: List[FeatureData]):
        """
        :param feature_data_list: List of feature data objects that should be (virtually) concatenated.
        """
        sample_sizes = [fd.get_n_samples() for fd in feature_data_list]
        super().__init__(n_samples=sum(sample_sizes), device=feature_data_list[0].get_device(),
                         dtype=feature_data_list[0].get_dtype())
        self.feature_data_list = feature_data_list
        self.sample_sizes = sample_sizes

    def iterate(self, idxs: Indexes) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        for i, sub_idxs in idxs.split_by_sizes(self.sample_sizes):
            for sub_idxs_2, feature_data in self.feature_data_list[i].iterate(sub_idxs):
                yield sub_idxs_2, feature_data

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        simplified = [self.feature_data_list[i].simplify(sub_idxs)
                      for i, sub_idxs in idxs.split_by_sizes(self.sample_sizes)]
        simplified = [fd for fd in simplified if not isinstance(fd, EmptyFeatureData)]
        if len(simplified) == 1:
            return simplified[0]
        elif len(simplified) == 0:
            return EmptyFeatureData(device=self.device)
        else:
            if utils.all_equal([type(fd) for fd in simplified]):
                # call simplify_multi_ of the sub-data
                # for example, if simplified[0] is a ListFeatureData object,
                # it will need to concatenate list elements separately
                return simplified[0].simplify_multi_(simplified, [Indexes(fd.get_n_samples(), None) for fd in simplified])
            else:
                # this should not occur in practice
                # because all concatenated FeatureData objects should have the same basic type
                raise RuntimeError(
                    'Attempting to concatenate different-typed simplified FeatureData objects during simplify')

    def simplify_multi_(self, feature_data_list: List['ConcatFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        if len(feature_data_list) == 0:
            return EmptyFeatureData(device=self.device)
        # use simplify() to implement simplify_multi_()
        return ConcatFeatureData([SubsetFeatureData(fd, idxs) for cfd, idxs in zip(feature_data_list, idxs_list)
                                  for fd in cfd.feature_data_list]).simplify()

    def cast_to(self, dtype) -> 'FeatureData':
        return ConcatFeatureData([fd.cast_to(dtype) for fd in self.feature_data_list])


class BatchedFeatureData(FeatureData):
    """
    This class can be used to force batched processing of data,
    by iterating over batches in its .iterate() and .__iter__() methods.
    In contrast to ConcatFeatureData, it does only pretend that it contains multiple sub FeatureData objects.
    Since it only holds one FeatureData object, it is more efficient to use for simplify() and get_tensor().
    """
    def __init__(self, feature_data: FeatureData, batch_size: int):
        """
        :param feature_data: FeatureData to be batched.
        :param batch_size: Batch size. The last batch is smaller
        if len(feature_data) is not divisible by batch_size.
        """
        super().__init__(n_samples=len(feature_data), device=feature_data.get_device(), dtype=feature_data.get_dtype())
        self.feature_data = feature_data
        self.batch_size = batch_size
        batch_intervals = utils.get_batch_intervals(self.n_samples, batch_size=batch_size)
        self.batch_idxs = [Indexes(self.n_samples, slice(start, stop)) for start, stop in batch_intervals]
        self.sample_sizes = [len(idxs) for idxs in self.batch_idxs]

    def iterate(self, idxs: Indexes) -> Iterable[Tuple[Indexes, 'FeatureData']]:
        # print(f'{idxs.get_idxs()=}, {list(idxs.split_by_sizes(self.sample_sizes))=}, {self.sample_sizes=}')
        # print(f'{idxs.get_idxs()=}, {self.sample_sizes=}')
        for i, sub_idxs in idxs.split_by_sizes(self.sample_sizes):
            for sub_idxs_2, feature_data in self.feature_data.iterate(self.batch_idxs[i].compose(sub_idxs)):
                yield sub_idxs_2, feature_data

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return self.feature_data.simplify(idxs)

    def simplify_multi_(self, feature_data_list: List['BatchedFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        if len(feature_data_list) == 0:
            return EmptyFeatureData(device=self.device)
        fd_list = [fd.feature_data for fd in feature_data_list]
        return fd_list[0].simplify_multi_(fd_list, idxs_list)

    def cast_to(self, dtype) -> 'FeatureData':
        return BatchedFeatureData(self.feature_data.cast_to(dtype), self.batch_size)


class ListFeatureData(FeatureData):  # does not concatenate along batch dimension
    """
    This class represents a list of separate FeatureData objects.
    All FeatureData objects must have the same number of samples.
    This is different to ConcatFeatureData,
    which assumes that the FeatureData objects are concatenated along the samples dimension.
    """
    def __init__(self, feature_data_list: List[FeatureData]):
        """
        :param feature_data_list: List of feature data objects.
        """
        super().__init__(n_samples=feature_data_list[0].get_n_samples(), device=feature_data_list[0].get_device(),
                         dtype=feature_data_list[0].get_dtype())
        self.feature_data_list = feature_data_list

    def get_tensor(self, idxs: Optional[Union[torch.Tensor, slice, int, Indexes]] = None) -> torch.Tensor:
        raise NotImplementedError(
            'get_tensor() cannot be called on ListFeatureData since it would need to return multiple tensors')

    def simplify_impl_(self, idxs: Indexes) -> 'FeatureData':
        return ListFeatureData([fd.simplify(idxs) for fd in self.feature_data_list])

    def simplify_multi_(self, feature_data_list: List['ListFeatureData'], idxs_list: List[Indexes]) -> 'FeatureData':
        if len(feature_data_list) == 0:
            return EmptyFeatureData(device=self.device, dtype=self.dtype)
        return ListFeatureData([ConcatFeatureData([fd.feature_data_list[i][idxs]
                                                   for fd, idxs in zip(feature_data_list, idxs_list)]).simplify()
                                for i in range(len(feature_data_list[0].feature_data_list))])

    def cast_to(self, dtype) -> 'FeatureData':
        return ListFeatureData([fd.cast_to(dtype) for fd in self.feature_data_list])
