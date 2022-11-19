import math

from .. import utils
from .feature_data import *


def robust_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """
    Implements a Cholesky decomposition.
    If the Cholesky decomposition fails, it is retried with increasing added jitter on the diagonal.
    :param matrix: Symmetric positive semi-definite matrix to factorize.
    :return: Approximate cholesky factor L such that (approximately) LL^T = matrix
    """
    eps = 1e-5 * matrix.trace() / matrix.shape[-1]
    L = None
    for i in range(10):
        try:
            L = torch.linalg.cholesky(matrix)
            break
        except RuntimeError:
            print('Increasing jitter for Cholesky decomposition', flush=True)
            matrix += eps * torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
            eps *= 2
    if L is None:
        raise RuntimeError('Could not Cholesky decompose the matrix')
    return L


def robust_cholesky_inv(matrix: torch.Tensor) -> torch.Tensor:
    """
    :param matrix: Symmetric positive semi-definite matrix.
    :return: A matrix A such that (approximately) matrix^{-1} = A^T A, where A = L^{-1} for the Cholesky factor L.
    """
    # returns a matrix A such that matrix^{-1} = A^T A
    L = robust_cholesky(matrix)
    # there might be a better solution, but I found no direct triangular inversion function
    return L.inverse()


class DataTransform:
    """
    Abstract base class for representing functions that transform FeatureData objects into other FeatureData objects,
    usually by applying the same function to each sample in the FeatureData object.
    This is typically used for (parts of) feature maps, except that it is more general than a feature map
    since it does not need to allow the computation of (inner products of) vector-valued features.
    For example, in a feature map phi(x) = f(g(x)), f is also a feature map, but g only needs to be a DataTransform.
    """
    def __call__(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> FeatureData:
        """
        Method to apply the transformation to (a subset of) the given feature data.
        Subclasses should not override this method, but override forward() instead.
        :param feature_data: FeatureData to apply this transformation to.
        :param idxs: An object to index the feature data with, or None if all of the feature data should be used.
        :return: Returns a FeatureData object.
        """
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        pieces = [self.forward(sub_data, sub_idxs)
                  for sub_idxs, sub_data in feature_data.iterate(idxs)]
        return ConcatFeatureData(pieces) if len(pieces) != 1 else pieces[0]

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        """
        Internal function for subclasses to override.
        :param feature_data: FeatureData to apply the transformation to.
        :param idxs: Indexes object corresponding to the subset of the feature data that should be processed.
        :return: Transformed FeatureData.
        """
        raise NotImplementedError()


class FeatureMap(DataTransform):
    """
    Abstract base class for representing feature maps and their corresponding kernel.
    For kernels with infinite-dimensional feature space,
    the corresponding sub-classes may only allow evaluation of the kernel and not of the feature map.
    """
    def __init__(self, n_features: int, allow_precompute_features: bool = True):
        """
        :param n_features: Feature space dimension of the feature map.
        Should be -1 if the feature space dimension is infinite.
        :param allow_precompute_features: Specifies whether the default behavior of precompute()
        should be to simply precompute the feature matrix.
        This might not be desirable for kernels with high feature-space dimension
        where more efficient kernel evaluation methods than the inner product between feature vectors exist.
        """
        self.n_features = n_features
        self.allow_precompute_features = allow_precompute_features

    def get_n_features(self):
        """
        :return: Returns the feature space dimension.
        """
        return self.n_features

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        """
        Implements the forward() method from DataTransform,
        which in this case computes the feature matrix wrapped in TensorFeatureData.
        This method can only be called if self.get_feature_matrix_impl_() is implemented.
        :param feature_data: Input data to the feature map.
        :param idxs: Indexes to index feature_data with.
        :return: Returns a TensorFeatureData object containing the feature matrix.
        """
        return TensorFeatureData(self.get_feature_matrix(feature_data, idxs))

    def precompute(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> Tuple['FeatureMap', FeatureData]:
        """
        Returns a tuple (fm, fd) such that the feature map fm on the feature data fd
        behaves identically to this feature map on feature_data, in terms of kernel matrix values.
        For example, this method may transform (phi, x) to (id, phi(x)), if self = phi and x = feature_data.
        The returned tuple should be at least as fast to evaluate as before.
        Subclasses should not override precompute() but precompute_soft_().
        :param feature_data: Input to this feature map.
        :param idxs: Indexes to index feature_data with, or None if the full data should be used.
        :return: Returns an equivalent tuple of FeatureMap and FeatureData.
        """
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        if self.allow_precompute_features:
            # simply compute the feature matrix and apply an identity feature map to it
            return IdentityFeatureMap(n_features=self.n_features), self(feature_data, idxs)
        else:
            # use another precomputation method that can be overridden by a subclass.
            results = [self.precompute_soft_(sub_data, sub_idxs) for sub_idxs, sub_data in feature_data.iterate(idxs)]
            if len(results) == 1:
                return results[0]
            # we do not call .simplify() on the feature data here
            # since precompute() might be called recursively on smaller parts of the data,
            # and we do not want to simplify() multiple times but only once after precompute()
            return results[0][0], ConcatFeatureData([r[1] for r in results])

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        """
        Internal method to precompute the feature map
        in the case that precomputing the whole feature matrix is not allowed.
        This method can be overridden by subclasses if needed.
        By default, it does not perform any precomputations and just returns (self, feature_data[idxs]).
        :param feature_data: Feature data to apply this feature map to.
        :param idxs: Indexes to index the feature data with.
        :return: Potentially precomputed tuple (FeatureMap, FeatureData).
        """
        # we do not call .simplify() on the feature data here
        # since precompute() might be called recursively on smaller parts of the data,
        # and we do not want to simplify() multiple times but only once after precompute()
        return self, feature_data[idxs]

    def posterior(self, feature_data: FeatureData, sigma: float, allow_kernel_space_posterior: bool = True) \
            -> 'FeatureMap':
        """
        Returns a feature map that represents the Gaussian Process posterior kernel after observing feature_data,
        if the noise variance is sigma^2.
        :param feature_data: Feature data to condition the Gaussian Process with.
        :param sigma: Noise standard deviation for the Gaussian Process.
        :param allow_kernel_space_posterior: Whether the method is allowed to use the kernel-space posterior formula
        if it is deemed more efficient but not strictly necessary. The kernel-space posterior feature map returned
        will not allow a computation of the feature matrix, which could be detrimental
        for methods that want to operate in feature space.
        :return: Returns a feature map that represents the Gaussian Process posterior kernel after observing feature_data,
        if the noise variance is sigma^2.
        """
        if self.n_features < 0 or (allow_kernel_space_posterior and self.n_features > max(1024, 3 * len(feature_data))):
            # compute the posterior in kernel space
            return KernelSpacePosteriorFeatureMap(feature_map=self, cond_data=feature_data, sigma=sigma)

        feature_matrix = self.get_feature_matrix(feature_data)
        eye = torch.eye(self.n_features, device=feature_matrix.device, dtype=feature_matrix.dtype)
        cov_matrix = feature_matrix.t().matmul(feature_matrix) + (sigma ** 2) * eye

        return SequentialFeatureMap(LinearFeatureMap(sigma * robust_cholesky_inv(cov_matrix).t()), [self])

    def get_feature_matrix(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> torch.Tensor:
        """
        Returns the feature matrix obtained by applying this feature map to feature_data[idxs].
        This method can only be used if get_feature_matrix_impl_() is implemented,
        in particular it cannot be used if the feature space dimension is infinite.
        Subclasses should not override this method but get_feature_matrix_impl_() instead.
        :param feature_data: FeatureData to apply this feature map to.
        :param idxs: Indexes to index feature_data with, or None to use the full feature_data.
        :return: Feature matrix as torch.Tensor of shape n_samples x n_features
        """
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        return torch_cat([self.get_feature_matrix_impl_(sub_data, sub_idxs)
                          for sub_idxs, sub_data in feature_data.iterate(idxs)], dim=-2)

    def get_kernel_matrix(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                          idxs_1: Optional[Indexes] = None, idxs_2: Optional[Indexes] = None) -> torch.Tensor:
        """
        Returns the kernel matrix k(feature_data_1[idxs_1], feature_data_2[idxs_2]),
        where k is the kernel corresponding to the current feature map.
        Subclasses should not override this method but get_kernel_matrix_impl_() instead.
        :param feature_data_1: First feature data.
        :param feature_data_2: Second feature data.
        :param idxs_1: Indexes for first feature data, or None.
        :param idxs_2: Indexes for second feature data, or None.
        :return: Kernel matrix as torch.Tensor of shape n_samples_1 x n_samples_2
        """
        idxs_1 = Indexes(feature_data_1.get_n_samples(), idxs_1)
        idxs_2 = Indexes(feature_data_2.get_n_samples(), idxs_2)
        return torch_cat([torch_cat([self.get_kernel_matrix_impl_(sub_data_1, sub_data_2, sub_idxs_1, sub_idxs_2)
                                     for sub_idxs_2, sub_data_2 in feature_data_2.iterate(idxs_2)], dim=-1)
                          for sub_idxs_1, sub_data_1 in feature_data_1.iterate(idxs_1)], dim=-2)

    def get_kernel_matrix_diag(self, feature_data: FeatureData, idxs: Optional[Indexes] = None) -> torch.Tensor:
        """
        Returns the diagonal of the kernel matrix k(feature_data[idxs], feature_data[idxs]),
        where k is the kernel corresponding to the current feature map.
        Subclasses should not override this method but get_kernel_matrix_diag_impl_() instead.
        :param feature_data: Feature data.
        :param idxs: Indexes to index the first feature data, or None.
        :return: Diagonal of Kernel matrix as torch.Tensor of shape [n_samples]
        """
        idxs = Indexes(feature_data.get_n_samples(), idxs)
        return torch_cat([self.get_kernel_matrix_diag_impl_(sub_data, sub_idxs)
                          for sub_idxs, sub_data in feature_data.iterate(idxs)], dim=-1)

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        """
        Internal method that should be overridden by subclasses to compute the feature matrix.
        :param feature_data: Feature data to apply this feature map to.
        :param idxs: Indexes to index feature_data.
        :return: Feature matrix as torch.tensor of shape n_samples x n_features
        """
        raise NotImplementedError()

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        """
        Internal method to compute the kernel matrix.
        By default, this method uses get_feature_matrix()
        and then computes a matrix-matrix product of two feature matrices.
        Subclasses should override this method if the kernel matrix should be computed in another way.
        :param feature_data_1: First feature data.
        :param feature_data_2: Second feature data.
        :param idxs_1: Indexes for the first feature data.
        :param idxs_2: Indexes for the second feature data.
        :return: Kernel matrix as torch.Tensor of shape n_samples_1 x n_samples_2
        """
        return self.get_feature_matrix(feature_data_1, idxs_1).matmul(
            self.get_feature_matrix(feature_data_2, idxs_2).t())

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        """
        Internal method to compute the diagonal of the kernel matrix.
        By default, this method uses the squared norm of the feature vectors obtained by get_feature_matrix().
        Subclasses should override this method if another computation should be used.
        :param feature_data: Feature data.
        :param idxs: Indexes to index the feature data.
        :return: Kernel matrix diagonal as torch.Tensor of shape [n_samples]
        """
        return (self.get_feature_matrix(feature_data, idxs) ** 2).sum(dim=-1)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        """
        Apply sketching (a.k.a. random projections) to this feature map,
        returning a feature map with (lower) feature-space dimension given by n_features.
        This method may not be implemented by all subclasses.
        :param n_features: Number of target features for the sketched feature map.
        :param config: This can be used to pass other parameters to individual sketching operations.
        :return: Returns a sketched feature map. Note that this does not modify any data yet,
        for this the sketched feature map needs to be precomputed on the data.
        """
        raise NotImplementedError()


class IdentityFeatureMap(FeatureMap):
    """
    This class represents the identity feature map phi(x) = x, and the linear kernel k(x, y) = x^T y.
    """
    def __init__(self, n_features: int):
        """
        :param n_features: Dimension of the inputs.
        """
        super().__init__(n_features=n_features)

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        return feature_data.get_tensor(idxs)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        # Gaussian sketch
        matrix = torch.randn(self.n_features, n_features)
        if config.get('sketch_norm', False):
            # it is possible to normalize the Gaussian vectors, which can make a difference for small input dimensions.
            matrix /= (matrix ** 2).sum(dim=0, keepdim=True).sqrt() / math.sqrt(self.n_features)
        matrix /= math.sqrt(n_features)
        return SequentialFeatureMap(LinearFeatureMap(matrix), [self])


class ReLUNTKFeatureMap(FeatureMap):
    """
    This feature map represents the Neural Tangent Kernel (Jacot et al., 2018) corresponding to a ReLU NN
    in Neural Tangent Parameterization. We implement the form of Lee et al. (2019), with factors sigma_w and sigma_b,
    and where the biases are initialized from N(0, 1).
    """
    # following SM 3 and 5 in
    # https://proceedings.neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html
    def __init__(self, n_layers=3, sigma_w_sq=2.0, sigma_b_sq=0.0):
        """
        :param n_layers: Number of layers of the corresponding NN
        :param sigma_w_sq: sigma_w**2 in the notation of Lee et al. (2019)
        :param sigma_b_sq: sigma_b**2 in the notation of Lee et al. (2019)
        """
        super().__init__(n_features=-1, allow_precompute_features=False)
        self.n_layers = n_layers
        self.sigma_w_sq = sigma_w_sq
        self.sigma_b_sq = sigma_b_sq

    def t_and_tdot_(self, a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_sqrt = a.sqrt()
        d_sqrt = d.sqrt()
        cos_theta = torch.clip(b / (a_sqrt * d_sqrt + 1e-30), min=-1.0, max=1.0)
        theta = torch.arccos(cos_theta)
        t = 1 / (2 * math.pi) * a_sqrt * d_sqrt * ((1 - cos_theta ** 2).sqrt() + (math.pi - theta) * cos_theta)
        tdot = 1 / (2 * math.pi) * (math.pi - theta)
        return t, tdot

    def diag_prop_(self, diag: torch.Tensor) -> torch.Tensor:
        # evaluate (k_l(x, x))_{x \in X} from diag = (k_{l-1}(x, x))_{x \in X}
        t = 0.5 * diag
        return self.sigma_w_sq * t + self.sigma_b_sq

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        feature_mat_1 = feature_data_1.get_tensor(idxs_1)
        feature_mat_2 = feature_data_2.get_tensor(idxs_2)
        kernel_mat = feature_mat_1.matmul(feature_mat_2.t())
        d_in = feature_mat_1.shape[1]
        diag_1 = self.sigma_w_sq / d_in * (feature_mat_1 ** 2).sum(dim=1) + self.sigma_b_sq
        diag_2 = self.sigma_w_sq / d_in * (feature_mat_2 ** 2).sum(dim=1) + self.sigma_b_sq
        nngp_mat = self.sigma_w_sq / d_in * kernel_mat + self.sigma_b_sq
        ntk_mat = nngp_mat
        for i in range(self.n_layers - 1):
            t, tdot = self.t_and_tdot_(diag_1[:, None], nngp_mat, diag_2[None, :])
            nngp_mat = self.sigma_w_sq * t + self.sigma_b_sq
            ntk_mat = nngp_mat + self.sigma_w_sq * ntk_mat * tdot
            diag_1 = self.diag_prop_(diag_1)
            diag_2 = self.diag_prop_(diag_2)
        return ntk_mat

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        feature_mat = feature_data.get_tensor(idxs)
        d_in = feature_mat.shape[1]
        nngp_diag = self.sigma_w_sq / d_in * (feature_mat ** 2).sum(dim=1) + self.sigma_b_sq
        ntk_diag = nngp_diag
        for i in range(self.n_layers - 1):
            nngp_diag = self.diag_prop_(nngp_diag)
            ntk_diag = nngp_diag + self.sigma_w_sq * 0.5 * ntk_diag
        return ntk_diag


class ReLUNNGPFeatureMap(FeatureMap):
    """
    This feature map represents the NNGP Kernel (Jacot et al., 2018) corresponding to a ReLU NN
    in Neural Tangent Parameterization. We implement the form of Lee et al. (2019), with factors sigma_w and sigma_b,
    and where the biases are initialized from N(0, 1).
    """
    def __init__(self, n_layers=3, sigma_w_sq=2.0, sigma_b_sq=0.0):
        """
        :param n_layers: Number of layers of the corresponding NN
        :param sigma_w_sq: sigma_w**2 in the notation of Lee et al. (2019)
        :param sigma_b_sq: sigma_b**2 in the notation of Lee et al. (2019)
        """
        super().__init__(n_features=-1, allow_precompute_features=False)
        self.n_layers = n_layers
        self.sigma_w_sq = sigma_w_sq
        self.sigma_b_sq = sigma_b_sq

    def get_t_(self, a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        a_sqrt = a.sqrt()
        d_sqrt = d.sqrt()
        cos_theta = torch.clip(b / (a_sqrt * d_sqrt + 1e-30), min=-1.0, max=1.0)
        theta = torch.arccos(cos_theta)
        t = 1 / (2 * math.pi) * a_sqrt * d_sqrt * ((1 - cos_theta ** 2).sqrt() + (math.pi - theta) * cos_theta)
        return t

    def diag_prop_(self, diag: torch.Tensor) -> torch.Tensor:
        # evaluate (k_l(x, x))_{x \in X} from diag = (k_{l-1}(x, x))_{x \in X}
        t = 0.5 * diag
        return self.sigma_w_sq * t + self.sigma_b_sq

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        feature_mat_1 = feature_data_1.get_tensor(idxs_1)
        feature_mat_2 = feature_data_2.get_tensor(idxs_2)
        kernel_mat = feature_mat_1.matmul(feature_mat_2.t())
        d_in = feature_mat_1.shape[1]
        diag_1 = self.sigma_w_sq / d_in * (feature_mat_1 ** 2).sum(dim=1) + self.sigma_b_sq
        diag_2 = self.sigma_w_sq / d_in * (feature_mat_2 ** 2).sum(dim=1) + self.sigma_b_sq
        nngp_mat = self.sigma_w_sq / d_in * kernel_mat + self.sigma_b_sq
        for i in range(self.n_layers - 1):
            t = self.get_t_(diag_1[:, None], nngp_mat, diag_2[None, :])
            nngp_mat = self.sigma_w_sq * t + self.sigma_b_sq
            diag_1 = self.diag_prop_(diag_1)
            diag_2 = self.diag_prop_(diag_2)
        return nngp_mat

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        feature_mat = feature_data.get_tensor(idxs)
        d_in = feature_mat.shape[1]
        nngp_diag = self.sigma_w_sq / d_in * (feature_mat ** 2).sum(dim=1) + self.sigma_b_sq
        for i in range(self.n_layers - 1):
            nngp_diag = self.diag_prop_(nngp_diag)
        return nngp_diag


class LaplaceKernelFeatureMap(FeatureMap):
    """
    Implements the Laplace kernel, k(x, y) = exp(-scale*||x-y||).
    """
    def __init__(self, scale: float = 1.0):
        """
        :param scale: Scale parameter of the Laplace kernel. Larger scale yields a narrower kernel.
        """
        super().__init__(n_features=-1, allow_precompute_features=False)
        self.scale = scale

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        feature_mat_1 = feature_data_1.get_tensor(idxs_1)
        feature_mat_2 = feature_data_2.get_tensor(idxs_2)

        sq_dist_mat = (feature_mat_1**2).sum(dim=-1)[:, None] + (feature_mat_2**2).sum(dim=-1)[None, :] \
                   - 2 * feature_mat_1 @ feature_mat_2.t()
        return torch.exp(-self.scale*torch.sqrt(sq_dist_mat))

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        feature_mat = feature_data.get_tensor(idxs)
        return torch.ones_like(feature_mat[:, 0])


class KernelSpacePosteriorFeatureMap(FeatureMap):
    """
    This internal class represents the posterior kernel of a GP after observing data.
    This is used internally by FeatureMap.posterior().
    """
    def __init__(self, feature_map: FeatureMap, cond_data: FeatureData, sigma: float):
        """
        :param feature_map: Prior feature map.
        :param cond_data: Data that the GP is conditioned on.
        :param sigma: Noise standard deviation of the GP.
        """
        super().__init__(n_features=-1, allow_precompute_features=False)
        mat = feature_map.get_kernel_matrix(cond_data, cond_data)
        # (K + sigma*I)^{-1} = L_inv.t() @ L_inv
        eye = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
        self.L_inv = robust_cholesky_inv(mat + sigma**2 * eye)
        self.inv_mat = self.L_inv.t() @ self.L_inv
        self.feature_map = feature_map
        self.cond_data = cond_data
        self.sigma = sigma

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        # Precomputation works as follows: The posterior kernel k_post is given by
        # k_post(x_1, x_2) = k(x_1, x_2) - k(x_1, X) (k(X, X) + sigma^2 I)^{-1} k(X, x_2) = k(x_1, x_2) - z_1^T z_2,
        # where z_i = (k(X, X) + sigma^2 I)^{-1/2} k(X, x_i).
        # Precomputation precomputes the z_i and appends them to the x_i via ListFeatureData.
        tensor = self.feature_map.get_kernel_matrix(feature_data, self.cond_data, idxs) @ self.L_inv.t()
        return PrecomputedKernelSpacePosteriorFeatureMap(self.feature_map), \
               ListFeatureData([feature_data[idxs].simplify(), TensorFeatureData(tensor)])

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        prior_mat = self.feature_map.get_kernel_matrix(feature_data_1, feature_data_2, idxs_1, idxs_2)
        mat_1 = self.feature_map.get_kernel_matrix(feature_data_1, self.cond_data, idxs_1)
        mat_2 = self.feature_map.get_kernel_matrix(feature_data_2, self.cond_data, idxs_2)
        return prior_mat - mat_1 @ self.inv_mat @ mat_2.t()

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        prior_diag = self.feature_map.get_kernel_matrix_diag(feature_data, idxs)
        mat = self.feature_map.get_kernel_matrix(feature_data, self.cond_data, idxs)
        return prior_diag - ((mat @ self.L_inv.t()) ** 2).sum(dim=1)


class PrecomputedKernelSpacePosteriorFeatureMap(FeatureMap):
    """
    Internal class arising from precomputing on a KernelSpacePosteriorFeatureMap.
    """
    def __init__(self, feature_map: FeatureMap):
        """
        :param feature_map: Prior feature map, which is applied to the non-precomputed part of the data.
        """
        super().__init__(n_features=-1, allow_precompute_features=False)
        self.feature_map = feature_map

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        if not isinstance(feature_data_1, ListFeatureData):
            raise ValueError(f'feature_data_1 must be of type ListFeatureData, but is of type {type(feature_data_1)}')
        if not isinstance(feature_data_2, ListFeatureData):
            raise ValueError(f'feature_data_2 must be of type ListFeatureData, but is of type {type(feature_data_2)}')
        prior_mat = self.feature_map.get_kernel_matrix(feature_data_1.feature_data_list[0],
                                                       feature_data_2.feature_data_list[0], idxs_1, idxs_2)
        tensor_1 = feature_data_1.feature_data_list[1].get_tensor(idxs_1)
        tensor_2 = feature_data_2.feature_data_list[1].get_tensor(idxs_2)
        return prior_mat - tensor_1 @ tensor_2.t()

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        prior_diag = self.feature_map.get_kernel_matrix_diag(feature_data.feature_data_list[0], idxs)
        tensor = feature_data.feature_data_list[1].get_tensor(idxs)
        return prior_diag - (tensor ** 2).sum(dim=1)


class ScaledFeatureMap(FeatureMap):
    """
    Represents a scaled feature map factor * feature_map.
    """
    def __init__(self, feature_map: FeatureMap, factor: float):
        """
        :param feature_map: Feature map to scale.
        :param factor: Scaling factor.
        """
        super().__init__(n_features=feature_map.get_n_features(),
                         allow_precompute_features=feature_map.allow_precompute_features)
        self.feature_map = feature_map
        self.factor = factor

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        fm, fd = self.feature_map.precompute(feature_data, idxs)
        return ScaledFeatureMap(fm, self.factor), fd

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        return self.factor * self.feature_map.get_feature_matrix(feature_data, idxs)

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        return self.factor ** 2 * self.feature_map.get_kernel_matrix(feature_data_1, feature_data_2, idxs_1, idxs_2)

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        return self.factor ** 2 * self.feature_map.get_kernel_matrix_diag(feature_data, idxs)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        return ScaledFeatureMap(self.feature_map.sketch(n_features, **config), self.factor)


class SumFeatureMap(FeatureMap):
    """
    Represents a feature map corresponding to the sum of multiple kernels.
    """
    def __init__(self, feature_maps: List[FeatureMap]):
        """
        :param feature_maps: Feature maps whose kernels should be added.
        """
        feature_counts = [fm.get_n_features() for fm in feature_maps]
        super().__init__(n_features=sum(feature_counts) if all([n >= 0 for n in feature_counts]) else -1,
                         allow_precompute_features=all([fm.allow_precompute_features for fm in feature_maps]))
        self.feature_maps = feature_maps

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        results = [fm.precompute(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)]
        return SumFeatureMap([r[0] for r in results]), ListFeatureData([r[1] for r in results])

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        return torch_cat(
            [fm.get_feature_matrix(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)],
            dim=-1)

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        if not isinstance(feature_data_1, ListFeatureData):
            raise ValueError(f'feature_data_1 must be of type ListFeatureData, but is of type {type(feature_data_1)}')
        if not isinstance(feature_data_2, ListFeatureData):
            raise ValueError(f'feature_data_2 must be of type ListFeatureData, but is of type {type(feature_data_2)}')
        return sum([fm.get_kernel_matrix(fd1, fd2, idxs_1, idxs_2)
                    for fm, fd1, fd2 in zip(self.feature_maps, feature_data_1.feature_data_list,
                                            feature_data_2.feature_data_list)])

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        return sum(
            [fm.get_kernel_matrix_diag(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)])

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        return ElementwiseSumFeatureMap([fm.sketch(n_features, **config) for fm in self.feature_maps])


class ProductFeatureMap(FeatureMap):
    """
    Feature map corresponding to the product of multiple kernels
    """
    def __init__(self, feature_maps: List[FeatureMap]):
        """
        :param feature_maps: Feature maps whose kernels should be multiplied.
        """
        # product features are too large, don't precompute in feature space
        feature_counts = [fm.get_n_features() for fm in feature_maps]
        has_finitely_many_features = all([n >= 0 for n in feature_counts])
        # allow precomp if all but one feature maps have only 1 feature, none has infinitely many,
        # and all can be precomputed
        # this can occur for the gradient feature map in the last layer of a NN
        allow_precompute_features = has_finitely_many_features \
                                    and (len([fc for fc in feature_counts if fc != 1]) <= 1) \
                                    and all([fm.allow_precompute_features for fm in feature_maps])
        super().__init__(n_features=utils.prod(feature_counts) if has_finitely_many_features else -1,
                         allow_precompute_features=allow_precompute_features)
        self.feature_maps = feature_maps

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        results = [fm.precompute(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)]
        return ProductFeatureMap([r[0] for r in results]), ListFeatureData([r[1] for r in results])

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        # raise RuntimeError('Computing feature matrix for product features!')
        slices = [[slice(None)] + [None] * i + [slice(None)] + [None] * (len(self.feature_maps) - 1 - i)
                  for i in range(len(self.feature_maps))]
        matrix_long = utils.prod([fm.get_feature_matrix(fd, idxs)[s]
                                  for fm, fd, s in zip(self.feature_maps, feature_data.feature_data_list, slices)])
        return matrix_long.reshape(matrix_long.shape[0], -1)

    def get_kernel_matrix_impl_(self, feature_data_1: FeatureData, feature_data_2: FeatureData,
                                idxs_1: Indexes, idxs_2: Indexes) -> torch.Tensor:
        if not isinstance(feature_data_1, ListFeatureData):
            raise ValueError(f'feature_data_1 must be of type ListFeatureData, but is of type {type(feature_data_1)}')
        if not isinstance(feature_data_2, ListFeatureData):
            raise ValueError(f'feature_data_2 must be of type ListFeatureData, but is of type {type(feature_data_2)}')
        return utils.prod([fm.get_kernel_matrix(fd1, fd2, idxs_1, idxs_2)
                           for fm, fd1, fd2 in zip(self.feature_maps, feature_data_1.feature_data_list,
                                                   feature_data_2.feature_data_list)])

    def get_kernel_matrix_diag_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        return utils.prod([fm.get_kernel_matrix_diag(fd, idxs)
                           for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)])

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        # use a simple tensor sketch, more complicated sketches could be used
        return ElementwiseProductFeatureMap([fm.sketch(n_features, **config) for fm in self.feature_maps])


class ElementwiseSumFeatureMap(FeatureMap):
    """
    Feature map representing the sum of multiple feature maps with equal feature space dimension.
    This is used for sketching SumFeatureMaps.
    """
    def __init__(self, feature_maps: List[FeatureMap]):
        """
        :param feature_maps: Feature maps whose features should be added.
        """
        super().__init__(n_features=feature_maps[0].get_n_features(),
                         allow_precompute_features=all([fm.allow_precompute_features for fm in feature_maps]))
        self.feature_maps = feature_maps

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        results = [fm.precompute(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)]
        return ElementwiseSumFeatureMap([r[0] for r in results]), ListFeatureData([r[1] for r in results])

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        return sum(
            [fm.get_feature_matrix(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)])


class ElementwiseProductFeatureMap(FeatureMap):
    """
    Feature map representing the elementwise product of multiple feature maps with equal feature space dimension.
    This is used for sketching ProductFeatureMaps.
    """
    def __init__(self, feature_maps: List[FeatureMap]):
        """
        :param feature_maps: Feature maps that should be multiplied element-wise.
        """
        super().__init__(n_features=feature_maps[0].get_n_features(),
                         allow_precompute_features=all([fm.allow_precompute_features for fm in feature_maps]))
        self.feature_maps = feature_maps

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        results = [fm.precompute(fd, idxs) for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)]
        return ElementwiseProductFeatureMap([r[0] for r in results]), ListFeatureData([r[1] for r in results])

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, ListFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        return self.n_features ** ((len(self.feature_maps) - 1) / 2) \
               * utils.prod([fm.get_feature_matrix(fd, idxs)
                             for fm, fd in zip(self.feature_maps, feature_data.feature_data_list)])


class LinearFeatureMap(FeatureMap):
    """
    Feature map of the form phi(x) = Ax for a matrix A.
    """
    def __init__(self, matrix: torch.Tensor):
        """
        :param matrix: should be in_features x out_features, i.e. transposed
        """
        super().__init__(n_features=matrix.shape[-1])
        self.matrix = matrix

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        if not isinstance(feature_data, TensorFeatureData):
            raise ValueError(f'feature_data must be of type ListFeatureData, but is of type {type(feature_data)}')
        feature_matrix = feature_data.get_tensor(idxs)
        self.matrix = self.matrix.to(feature_matrix.device)
        self.matrix = self.matrix.type(feature_matrix.type())
        return feature_matrix.matmul(self.matrix)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        # sketch by using the sketching function of IdentityFeatureMap
        return SequentialFeatureMap(
            IdentityFeatureMap(n_features=self.get_n_features()), [self]).sketch(n_features, **config)


class SequentialFeatureMap(FeatureMap):
    """
    Represents a feature map of the form phi(x) = f(g(x)) or even more concatenated functions.
    Here, g does not need to be a feature map but only a DataTransform.
    """
    def __init__(self, feature_map: FeatureMap, tfms: List[Callable]):
        """
        :param feature_map: Feature map that is applied last, after all the transforms.
        :param tfms: Transforms, in the order that they should be applied to the input.
        """
        super().__init__(n_features=feature_map.get_n_features(),
                         allow_precompute_features=feature_map.allow_precompute_features)
        self.feature_map = feature_map
        self.tfms = tfms

    def precompute_soft_(self, feature_data: FeatureData, idxs: Indexes) -> Tuple['FeatureMap', FeatureData]:
        feature_data = SubsetFeatureData(feature_data, idxs)
        for tfm in self.tfms:
            feature_data = tfm(feature_data)
        return self.feature_map.precompute(feature_data)

    def get_feature_matrix_impl_(self, feature_data: FeatureData, idxs: Indexes) -> torch.Tensor:
        feature_data = self.tfms[0](feature_data, idxs)  # or: SubsetFeatureData(feature_data, idxs)
        for tfm in self.tfms[1:]:
            feature_data = tfm(feature_data)
        return self.feature_map.get_feature_matrix(feature_data)

    def sketch(self, n_features: int, **config) -> 'FeatureMap':
        return SequentialFeatureMap(self.feature_map.sketch(n_features, **config), self.tfms)


class ToDoubleTransform(DataTransform):
    """
    Transforms data to float64 format
    """
    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        return SubsetFeatureData(feature_data.cast_to(torch.float64), idxs)
