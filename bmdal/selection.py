from abc import ABC
import numpy as np

import torch
from typing import *

from .. import utils
from .features import *


class SelectionMethod:
    """
    Abstract base class for selection methods,
    which allow to select a subset of indices from the pool set as the next batch to label for Batch Active Learning.
    """
    def __init__(self):
        super().__init__()
        self.status = None  # can be used to report errors during selection

    def select(self, batch_size: int) -> torch.Tensor:
        """
        Select batch_size elements from the pool set
        (which is assumed to be given in the constructor of the corresponding subclass).
        This method needs to be implemented by subclasses.
        It is assumed that this method is only called once per object, since it can modify the state of the object.
        :param batch_size: Number of elements to select from the pool set.
        :return: Returns a torch.Tensor of integer type that contains the selected indices.
        """
        raise NotImplementedError()

    def get_status(self) -> Optional:
        """
        :return: Returns an object representing the status of the selection. If all went well, the method returns None.
        Otherwise it might return a string or something different representing an error that occured.
        This is mainly useful for analyzing a lot of experiment runs.
        """
        return self.status


class IterativeSelectionMethod(SelectionMethod):
    """
    Abstract base class for iterative selection methods, as considered in the paper.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool, verbosity: int = 1,
                 **config):
        """
        :param pool_features: Features representing the pool set.
        :param train_features: Features representing the training set.
        :param sel_with_train: This corresponds to the mode parameter in the paper.
        Set to True if you want to use TP-mode (i.e. use the training data for selection), and to False for P-mode.
        :param verbosity: Level of verbosity. If >= 1, something may be printed to indicate the progress of selection.
        """
        super().__init__()
        self.train_features = train_features
        self.pool_features = pool_features
        self.features = pool_features.concat_with(train_features) if sel_with_train else pool_features
        self.selected_idxs = []
        self.selected_arr = torch.zeros(self.pool_features.get_n_samples(), dtype=torch.bool,
                                        device=self.pool_features.get_device())
        self.with_train = sel_with_train
        self.verbosity = verbosity

    def prepare(self, n_adds: int):
        """
        Callback method that may be implemented by subclasses.
        This method is called before starting the selection.
        It can be used, for example, to allocate memory depending on the batch size.
        :param n_adds: How often add() will be called during select()
        """
        pass

    def get_scores(self) -> torch.Tensor:
        """
        Abstract method that can be implemented by subclasses.
        This method should return a score for each pool sample,
        which can then be used to select the next pool sample to include.
        :return: Returns a torch.Tensor of shape [len(self.pool_features)] containing the scores.
        """
        raise NotImplementedError()

    def add(self, new_idx: int):
        """
        Update the state of the object (and therefore the scores) based on adding a new point to the selected set.
        :param new_idx: idx of the new point wrt to self.features,
        i.e. if new_idx > len(self.pool_features),
        then new_idx - len(self.pool_features) is the index for self.train_features,
        otherwise new_idx is an index to self.pool_features.
        """
        raise NotImplementedError()

    def get_next_idx(self) -> Optional[int]:
        """
        This method may be overridden by subclasses.
        It should return the index of the next sample that should be added to the batch.
        By default, it returns the index with the maximum score, according to self.get_scores().
        :return: Returns an int corresponding to the next index.
        It may also return None to indicate that the selection of the next index failed
        and that the batch should be filled up with random samples.
        """
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        return torch.argmax(self.get_scores()).item()

    def select(self, batch_size: int) -> torch.Tensor:
        """
        Iterative implementation of batch selection for Batch Active Learning.
        :param batch_size: Number of elements that should be included in the batch.
        :return: Returns a torch.Tensor of integer type containing the indices of the selected pool set elements.
        """
        device = self.pool_features.get_device()

        self.prepare(batch_size + len(self.train_features) if self.with_train else batch_size)

        if self.with_train:
            # add training points first
            for i in range(len(self.train_features)):
                self.add(len(self.pool_features)+i)
                if (i+1) % 256 == 0 and self.verbosity >= 1:
                    print(f'Added {i+1} train samples to selection', flush=True)

        for i in range(batch_size):
            next_idx = self.get_next_idx()
            if next_idx is None or next_idx < 0 or next_idx >= len(self.pool_features) or self.selected_arr[next_idx]:
                # data selection failed
                # fill up with random remaining indices
                self.status = f'filling up with random samples because selection failed after n_selected = {len(self.selected_idxs)}'
                if self.verbosity >= 1:
                    print(self.status)
                n_missing = batch_size - len(self.selected_idxs)
                remaining_idxs = torch.nonzero(~self.selected_arr).squeeze(-1)
                new_random_idxs = remaining_idxs[torch.randperm(len(remaining_idxs), device=device)[:n_missing]]
                selected_idxs_tensor = torch.as_tensor(self.selected_idxs, dtype=torch.long,
                                                       device=torch.device(device))
                return torch.cat([selected_idxs_tensor, new_random_idxs], dim=0)
            else:
                self.add(next_idx)
                self.selected_idxs.append(next_idx)
                self.selected_arr[next_idx] = True
        return torch.as_tensor(self.selected_idxs, dtype=torch.long, device=torch.device(device))


class ForwardBackwardSelectionMethod(IterativeSelectionMethod, ABC):
    """
    This class represents a forward-backward selection method that first selects too many samples
    and then removes some of them. It is an extension of the iterative selection method template.
    This is used for the forward-backward version of the BAIT selection method.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool, verbosity: int = 1,
                 overselection_factor: float = 1.0, **config):
        super().__init__(pool_features=pool_features, train_features=train_features, sel_with_train=sel_with_train,
                         verbosity=verbosity, **config)
        self.overselection_factor = overselection_factor

    def get_scores_backward(self) -> torch.Tensor:
        """
        Abstract method that can be implemented by subclasses.
        This method should return a score for each selected sample,
        which can then be used to select the next pool sample to include.
        :return: Returns a torch.Tensor of shape [len(self.selected_idxs)] containing the scores.
        """
        raise NotImplementedError()

    def get_next_idx_backward(self) -> Optional[int]:
        """
        This method may be overridden by subclasses.
        It should return the index of the next sample that should be removed from the batch.
        By default, it returns the index with the maximum score, according to self.get_scores_backward().
        :return: Returns an int corresponding to the next index to be removed.
        It may also return None to indicate that the selection of the next index failed
        and that the batch should be filled up with random samples.
        """
        scores = self.get_scores_backward().clone()
        return torch.argmax(scores).item()

    def remove(self, idx: int):
        """
        Update the state of the object (and therefore the scores) based on removing a point from the selected set.
        :param idx: index of the new point wrt to self.selected_idxs.
        """
        raise NotImplementedError()

    def select(self, batch_size: int) -> torch.Tensor:
        """
        Iterative implementation of batch selection for Batch Active Learning.
        :param batch_size: Number of elements that should be included in the batch.
        :return: Returns a torch.Tensor of integer type containing the indices of the selected pool set elements.
        """
        overselect_batch_size = min(round(self.overselection_factor * batch_size), len(self.pool_features))
        overselect_batch = super().select(overselect_batch_size)
        if batch_size == overselect_batch_size:
            return overselect_batch
        if self.status is not None:
            # selection failed
            return overselect_batch[:batch_size]

        for i in range(overselect_batch_size-batch_size):
            next_idx = self.get_next_idx_backward()
            if next_idx is None or next_idx < 0 or next_idx >= len(self.selected_idxs):
                self.status = f'removing the latest overselected samples because the backward step failed '\
                              f'after removing {i} samples'
                if self.verbosity >= 1:
                    print(self.status)
                self.selected_idxs = self.selected_idxs[:batch_size]
                break
            else:
                self.remove(next_idx)
                self.selected_arr[self.selected_idxs[next_idx]] = False
                del self.selected_idxs[next_idx]
        device = self.pool_features.get_device()
        return torch.as_tensor(self.selected_idxs, dtype=torch.long, device=torch.device(device))


class RandomSelectionMethod(SelectionMethod):
    """
    Implements random selection for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, **config):
        super().__init__()
        self.pool_features = pool_features

    def select(self, batch_size: int) -> torch.Tensor:
        device = self.pool_features.get_device()
        generator = torch.Generator(device=device)
        return torch.randperm(self.pool_features.get_n_samples(),
                              device=self.pool_features.get_device(),
                              generator=generator)[:batch_size]


class MaxDiagSelectionMethod(SelectionMethod):
    """
    Implements MaxDiag, i.e. naive active learning, for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, **config):
        """
        :param pool_features: Features on the pool set.
        """
        super().__init__()
        self.pool_features = pool_features

    def select(self, batch_size: int) -> torch.Tensor:
        diag = self.pool_features.get_kernel_matrix_diag()
        return torch.argsort(diag)[-batch_size:]


class MaxDetSelectionMethod(IterativeSelectionMethod):
    """
    Implements the kernel-space version of MaxDet for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 noise_sigma: float = 0.0, **config):
        """
        :param pool_features: Features on the pool set.
        :param train_features: Features on the train set.
        :param sel_with_train: If True, TP-mode is used instead of P-mode. This is False by default.
        :param noise_sigma: noise_sigma**2 is added to the kernel diagonal for determinant maximization.
        """
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.noise_sigma = noise_sigma
        self.diag = self.features.get_kernel_matrix_diag() + self.noise_sigma**2
        self.l = None
        self.n_added = 0

    def prepare(self, n_adds: int):
        #
        # usually self.l is None, since select() should only be called once
        n_total = n_adds if self.l is None else n_adds + self.l.shape[1]
        new_l = torch.zeros(len(self.features), n_total, device=torch.device(self.features.get_device()), dtype=self.diag.dtype)
        if self.l is not None:
            new_l[:, :self.l.shape[1]] = self.l
        self.l = new_l

    def get_scores(self) -> torch.Tensor:
        return self.diag

    def get_next_idx(self) -> Optional[int]:
        # print('max score:', torch.max(self.get_scores()).item())
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        new_idx = torch.argmax(scores).item()
        if scores[new_idx] <= 0.0:
            print(f'Selecting index {len(self.selected_idxs)+1}: new diag entry nonpositive')
            # print(f'diag entry: {self.get_scores()[new_idx]}')
            # diagonal is zero or lower, would cause numerical errors afterwards
            return None
        # print(self.diag[new_idx])
        return new_idx

    def add(self, new_idx: int):
        # print('new_idx:', new_idx)
        l = None if self.l is None else self.l[:, :self.n_added]
        lTl = 0.0 if l is None else l.matmul(l[new_idx, :])
        mat_col = self.features[new_idx].get_kernel_matrix(self.features).squeeze(0)
        if self.noise_sigma > 0.0:
            mat_col[new_idx] += self.noise_sigma**2
        update = (1.0 / torch.sqrt(self.diag[new_idx])) * (mat_col - lTl)
        # shape: len(self.features)
        self.diag -= update ** 2
        # shape: (n-1) x len(self.features)
        self.l[:, self.n_added] = update
        # self.l = update[:, None] if self.l is None else torch.cat([self.l, update[:, None]], dim=1)
        # print('trace(ll^T):', (self.l**2).sum())

        self.n_added += 1

        # if str(self.pool_features.get_device()) != 'cpu':
        #     torch.cuda.empty_cache()

        self.diag[new_idx] = -np.Inf   # ensure that the index is not selected again


class MaxDetFeatureSpaceSelectionMethod(IterativeSelectionMethod):
    """
    Implements MaxDet in feature space for Batch Active Learning.
    In terms of efficiency, this is only recommended for (roughly) batch_size > 3*n_features,
    so normally MaxDetSelectionMethod should be used instead.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 noise_sigma: float = 0.0, **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.noise_sigma = noise_sigma
        self.diag = self.features.get_kernel_matrix_diag().clone() + self.noise_sigma**2
        self.feature_matrix = self.features.get_feature_matrix().clone()

    def get_scores(self) -> torch.Tensor:
        return self.diag

    def get_next_idx(self) -> Optional[int]:
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        new_idx = torch.argmax(scores).item()
        if scores[new_idx] <= 0.0:
            if self.verbosity >= 1:
                print(f'Selecting index {len(self.selected_idxs)+1}: new diag entry nonpositive')
            # diagonal is zero or lower, would cause numerical errors afterwards
            return None
        return new_idx

    def add(self, new_idx: int):
        diag_entry = self.diag[new_idx]
        sqrt_diag_entry = torch.sqrt(diag_entry)
        beta = 1.0 / (sqrt_diag_entry * (sqrt_diag_entry + self.noise_sigma))
        phi_x = self.feature_matrix[new_idx]
        dot_prods = self.feature_matrix.matmul(phi_x)
        self.feature_matrix -= dot_prods[:, None] * (beta * phi_x[None, :])
        self.diag -= dot_prods**2 / diag_entry
        
        
class BaitFeatureSpaceSelectionMethod(ForwardBackwardSelectionMethod):
    """
    Implements BAIT in feature space for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 noise_sigma: float = 0.0, **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.noise_sigma = noise_sigma
        self.diag = self.features.get_kernel_matrix_diag().clone()
        self.feature_matrix = self.features.get_feature_matrix().clone()

        self.feature_cov_matrix = self.feature_matrix.t() @ self.feature_matrix
        if not sel_with_train:
            train_feature_matrix = train_features.get_feature_matrix()
            self.feature_cov_matrix += train_feature_matrix.t() @ train_feature_matrix
        self.scores_numerator = torch.einsum('ij,ji->i',
                                             self.feature_matrix,
                                             self.feature_cov_matrix @ self.feature_matrix.t())

    def get_scores(self) -> torch.Tensor:
        return self.scores_numerator / (self.diag + self.noise_sigma**2 + 1e-8)

    def get_next_idx(self) -> Optional[int]:
        scores = self.get_scores()
        scores[self.selected_idxs] = -np.Inf
        new_idx = torch.argmax(scores).item()
        if scores[new_idx].item() <= 0.0:
            if self.verbosity >= 1:
                print(f'Selecting index {len(self.selected_idxs)+1}: new score nonpositive')
            return None
        return new_idx

    def add(self, new_idx: int):
        diag_entry = self.diag[new_idx] + self.noise_sigma**2
        sqrt_diag_entry = torch.sqrt(diag_entry)
        beta = 1.0 / (sqrt_diag_entry * (sqrt_diag_entry + self.noise_sigma))
        phi_x = self.feature_matrix[new_idx]
        dot_prods = self.feature_matrix.matmul(phi_x)
        dot_prods_sq = dot_prods**2

        # update scores_numerator
        cov_phi = self.feature_cov_matrix @ phi_x
        phi_cov_phi = self.scores_numerator[new_idx].clone()
        # phi_cov_phi = torch.dot(phi_x, cov_phi)
        mult = 1/diag_entry
        self.scores_numerator -= 2 * mult * (self.feature_matrix @ cov_phi) * dot_prods
        self.scores_numerator += mult**2 * phi_cov_phi * dot_prods_sq
        # update feature_cov_matrix
        cov_phi_phit = cov_phi[:, None] * phi_x[None, :]
        phi_phit = phi_x[:, None] * phi_x[None, :]
        self.feature_cov_matrix -= beta * (cov_phi_phit + cov_phi_phit.t())
        self.feature_cov_matrix += beta**2 * phi_cov_phi * phi_phit
        # update feature matrix
        self.feature_matrix -= dot_prods[:, None] * (beta * phi_x[None, :])
        # update diag
        self.diag -= dot_prods_sq / diag_entry


    def get_scores_backward(self) -> torch.Tensor:
        den = (self.diag[self.selected_idxs] - self.noise_sigma ** 2)
        num = torch.clamp(self.scores_numerator[self.selected_idxs], min=0.0)
        scores = num / den
        scores[den >= 0.0] = -np.Inf
        return scores

    def get_next_idx_backward(self) -> Optional[int]:
        scores = self.get_scores_backward()
        new_idx = torch.argmax(scores).item()
        new_score = scores[new_idx].item()
        if new_score == -np.Inf or new_score >= 0.0:
            if self.verbosity >= 1:
                print(f'Backwards selecting index {len(self.selected_idxs)}: new score positive')
            return None
        return new_idx

    def remove(self, idx: int):
        features_idx = self.selected_idxs[idx]
        diag_entry = self.noise_sigma ** 2 - self.diag[features_idx]
        diag_entry = torch.clamp(diag_entry, min=1e-15)
        sqrt_diag_entry = torch.sqrt(diag_entry)
        beta = 1.0 / (sqrt_diag_entry * (sqrt_diag_entry + self.noise_sigma))
        phi_x = self.feature_matrix[features_idx]
        dot_prods = self.feature_matrix.matmul(phi_x)
        dot_prods_sq = dot_prods**2

        # update scores_numerator
        cov_phi = self.feature_cov_matrix @ phi_x
        phi_cov_phi = self.scores_numerator[features_idx].clone()
        mult = 1 / diag_entry
        self.scores_numerator += 2 * mult * (self.feature_matrix @ cov_phi) * dot_prods
        self.scores_numerator += mult ** 2 * phi_cov_phi * dot_prods_sq
        # update feature_cov_matrix
        cov_phi_phit = cov_phi[:, None] * phi_x[None, :]
        phi_phit = phi_x[:, None] * phi_x[None, :]
        self.feature_cov_matrix += beta * (cov_phi_phit + cov_phi_phit.t())
        self.feature_cov_matrix += beta ** 2 * phi_cov_phi * phi_phit
        # update feature matrix
        self.feature_matrix += dot_prods[:, None] * (beta * phi_x[None, :])
        # update diag
        self.diag += dot_prods_sq / diag_entry


class MaxDistSelectionMethod(IterativeSelectionMethod):
    """
    Implements the MaxDist selection method for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = True,
                 **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.min_sq_dists = np.Inf * torch.ones(self.pool_features.get_n_samples(), device=pool_features.get_device())

    def get_scores(self) -> torch.Tensor:
        return self.min_sq_dists

    def get_next_idx(self) -> Optional[int]:
        if len(self.selected_idxs) == 0:
            # no point added yet, take point with largest norm
            return torch.argmax(self.pool_features.get_kernel_matrix_diag()).item()
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        idx = torch.argmax(scores).item()
        # print('Next idx:', idx, '- Value:', scores[idx].item())
        return idx

    def add(self, new_idx: int):
        sq_dists = self.features[new_idx].get_sq_dists(self.pool_features).squeeze(0)
        self.min_sq_dists = torch.minimum(self.min_sq_dists, sq_dists)
        # print('min_sq_dists:', self.min_sq_dists)
        # if new_idx < len(self.pool_features):
        #     print('sq dists at new idx:', sq_dists[new_idx].item(), 'and', self.min_sq_dists[new_idx].item())


class LargestClusterMaxDistSelectionMethod(IterativeSelectionMethod):
    """
    Implements the LCMD selection method for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = True,
                 dist_weight_mode: str = 'sq-dist', **config):
        """
        :param pool_features:
        :param train_features:
        :param sel_with_train:
        :param dist_weight_mode: one of 'none', 'dist' or 'sq-dist'
        """
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.dist_weight_mode = dist_weight_mode
        self.min_sq_dists = np.Inf * torch.ones(self.pool_features.get_n_samples(), dtype=pool_features.get_dtype(),
                                                device=pool_features.get_device())
        self.closest_idxs = torch.zeros(self.pool_features.get_n_samples(), device=pool_features.get_device(),
                                        dtype=torch.long)
        self.neg_inf_tensor = torch.as_tensor([-np.Inf], dtype=pool_features.get_dtype(),
                                              device=pool_features.get_device())

        self.n_added = 0

    def get_scores(self) -> torch.Tensor:
        if self.dist_weight_mode == 'sq-dist':
            weights = self.min_sq_dists
        elif self.dist_weight_mode == 'dist':
            weights = self.min_sq_dists.sqrt()
        else:
            weights = None
        bincount = torch.bincount(self.closest_idxs, weights=weights, minlength=self.n_added+1)
        max_bincount = torch.max(bincount)
        # print(f'max bincount: {max_bincount.item():g}, max_min_sq_dist: {torch.max(self.min_sq_dists).item():g}, '
        #       f'number of zero dists: {self.min_sq_dists.shape[0] - torch.count_nonzero(self.min_sq_dists).item()}, ',
        #       f'average kernel diag: {self.features.get_kernel_matrix_diag().mean().item():g}')
        return torch.where(bincount[self.closest_idxs] == max_bincount, self.min_sq_dists, self.neg_inf_tensor)

    def get_next_idx(self) -> Optional[int]:
        if len(self.selected_idxs) == 0:
            # no point added yet, take point with largest norm
            return torch.argmax(self.pool_features.get_kernel_matrix_diag()).item()
        scores = self.get_scores().clone()
        scores[self.selected_idxs] = -np.Inf
        idx = torch.argmax(scores).item()
        return idx

    def add(self, new_idx: int):
        sq_dists = self.features[new_idx].get_sq_dists(self.pool_features).squeeze(0)
        self.n_added += 1
        new_min = sq_dists < self.min_sq_dists
        self.closest_idxs[new_min] = self.n_added
        self.min_sq_dists[new_min] = sq_dists[new_min]


class FrankWolfeSelectionMethod(IterativeSelectionMethod):
    """
    Implements the FrankWolfe selection method in feature space for Batch Active Learning.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        if pool_features.get_n_features() <= 0:
            raise NotImplementedError('Frank-Wolfe is currently only implemented for finite-dimensional feature maps '
                                      'due to quadratic time in n_pool for infinite-dimensional feature maps')
        self.sqrt_diag = self.features.get_kernel_matrix_diag().sqrt()
        self.sqrt_diag_sum = self.sqrt_diag.sum()
        self.feature_matrix = self.features.get_feature_matrix()
        self.eps = 1e-30
        self.normalized_feature_matrix = self.feature_matrix / (self.sqrt_diag[:, None] + self.eps)
        self.kernel_sum_embedding = self.feature_matrix.sum(dim=0)
        self.current_embedding = torch.zeros(self.feature_matrix.shape[-1], device=self.feature_matrix.device)

    def get_scores(self) -> torch.Tensor:
        return self.normalized_feature_matrix.matmul(self.kernel_sum_embedding - self.current_embedding)

    def get_next_idx(self) -> Optional[int]:
        scores = self.get_scores()
        scores[self.selected_idxs] = -np.Inf
        return torch.argmax(scores).item()

    def add(self, new_idx: int):
        normalized_feature_vector = self.normalized_feature_matrix[new_idx, :]
        residual = self.kernel_sum_embedding - self.current_embedding
        current_diff = self.sqrt_diag_sum * normalized_feature_vector - self.current_embedding
        gamma = torch.dot(current_diff, residual) / torch.dot(current_diff, current_diff)
        self.current_embedding = (1 - gamma) * self.current_embedding \
                                 + gamma * self.sqrt_diag_sum * normalized_feature_vector


class FrankWolfeKernelSpaceSelectionMethod(IterativeSelectionMethod):
    """
    Implements the FrankWolfe selection method in kernel space for Batch Active Learning.
    Note that runtime of the kernel-space version here scales quadratically with the pool set size.
    Therefore, it should be avoided for large pool sets. The feature-space version, FrankWolfeSelectionMethod,
    scales better for moderately large feature space dimensions.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = False,
                 **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        if pool_features.get_n_features() <= 0:
            raise NotImplementedError('Frank-Wolfe is currently only implemented for finite-dimensional feature maps '
                                      'due to quadratic time in n_pool for infinite-dimensional feature maps')
        self.sqrt_diag = self.features.get_kernel_matrix_diag().sqrt()  # c
        self.sqrt_diag_sum = self.sqrt_diag.sum()  # r
        batch_size = 2**22 // len(self.features) + 1
        batch_intervals = utils.get_batch_intervals(len(self.features), batch_size=batch_size)
        self.u = torch_cat([self.features[start:stop].get_kernel_matrix(self.features).sum(dim=-1)
                            for start, stop in batch_intervals], dim=0)
        self.v = torch.zeros_like(self.u)
        self.s = 0.0
        self.t = 0.0
        self.eps = 1e-30

    def get_scores(self) -> torch.Tensor:
        return (self.u - self.v) / (self.sqrt_diag + self.eps)

    def get_next_idx(self) -> Optional[int]:
        scores = self.get_scores()
        scores[self.selected_idxs] = -np.Inf
        return torch.argmax(scores).item()

    def add(self, new_idx: int):
        rcinv = self.sqrt_diag_sum / self.sqrt_diag[new_idx]
        ui = self.u[new_idx]
        vi = self.v[new_idx]
        denominator = (self.sqrt_diag_sum**2 - 2*rcinv*vi + self.s)
        denominator = max(denominator, self.eps)
        gamma = (rcinv * (ui - vi) + self.s - self.t) / denominator
        self.s = (1-gamma)**2 * self.s + 2*(1-gamma)*gamma*rcinv*vi + gamma**2 * self.sqrt_diag_sum**2
        self.t = (1-gamma)*self.t + gamma * rcinv * ui
        self.v = (1-gamma)*self.v + gamma * rcinv * self.features.get_kernel_matrix(self.features[new_idx])[:, 0]


class KmeansppSelectionMethod(MaxDistSelectionMethod):
    """
    Implements the KMeansPP (k-means++) selection method for Batch Active Learning.
    This class inherits from MaxDistSelectionMethod to reuse the squared-distance computations.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = True,
                 **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)

    def get_next_idx(self) -> Optional[int]:
        if len(self.selected_idxs) == 0:
            return np.random.randint(low=0, high=len(self.pool_features))
        try:
            return torch.multinomial(torch.clamp(self.min_sq_dists, min=0.0), 1).item()
        except RuntimeError:
            return None  # failed, potentially because the sum of the dists was <= 0


class RandomizedMinDistSumSelectionMethod(MaxDistSelectionMethod):
    """
    Experimental selection method.
    """
    def __init__(self, pool_features: Features, train_features: Features, sel_with_train: bool = True,
                 max_n_candidates: int = 5, **config):
        super().__init__(pool_features=pool_features, train_features=train_features,
                         sel_with_train=sel_with_train, **config)
        self.max_n_candidates = max_n_candidates

    def get_next_idx(self) -> Optional[int]:
        n_candidates = min(self.max_n_candidates, len(self.pool_features) - len(self.selected_idxs))
        weights = torch.ones_like(self.min_sq_dists) if len(self.selected_idxs) == 0 \
            else torch.clamp(self.min_sq_dists, min=0.0)
        candidates = torch.multinomial(weights, n_candidates)
        sq_dists = self.pool_features[candidates].get_sq_dists(self.pool_features)
        new_sq_dist_sums = torch.minimum(self.min_sq_dists[None, :].expand(n_candidates, -1), sq_dists).sum(dim=-1)
        new_sq_dist_sums[self.selected_idxs] = np.Inf
        return candidates[torch.argmin(new_sq_dist_sums)].item()


class SumOfSquaredDistsSelectionMethod(MaxDistSelectionMethod):
    """
    Experimental selection method. Scales quadratically with the pool set size.
    """
    def get_scores(self) -> torch.Tensor:
        print('computing scores')
        comp_batch_size = 512
        batch_intervals = utils.get_batch_intervals(len(self.pool_features), batch_size=comp_batch_size)
        return -torch_cat([
            torch.minimum(self.min_sq_dists[None, :].expand(stop-start, -1),
                          self.pool_features[start:stop].get_sq_dists(self.pool_features)).sum(dim=1)
            for start, stop in batch_intervals
        ], dim=0)


