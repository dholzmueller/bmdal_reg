from .feature_maps import *


class Features:
    """
    This class represents a combination of a feature map and feature data.
    Hence, it implicitly represents a (precomputed or non-precomputed) feature matrix.
    Whenever operations between to Features objects are used,
    it is assumed that both objects share the same feature map.
    Features objects can be transformed in various ways. Here is an example of a typical use-case:
    # initialize feature_map, train_feature_data, pool_feature_data
    train_features = Features(feature_map, train_feature_data)
    pool_features = Features(feature_map, pool_feature_data)
    tfm = train_features.posterior_tfm(sigma=0.1)
    post_train_features = tfm(train_features)
    post_pool_features = tfm(pool_features)
    post_pool_variances = post_pool_features.get_kernel_matrix_diag()
    """
    def __init__(self, feature_map: FeatureMap, feature_data: FeatureData, diag: Optional[torch.Tensor] = None):
        """
        :param feature_map: Feature map.
        :param feature_data: Data that serves as input to the feature map.
        :param diag: Optional parameter representing the precomputed kernel matrix diagonal, i.e.,
        feature_map.get_kernel_matrix_diag(feature_data).
        """
        self.feature_map = feature_map
        self.feature_data = feature_data
        self.diag = diag
        if diag is not None and not isinstance(diag, torch.Tensor):
            raise ValueError(f'diag has wrong type {type(diag)}')

    def precompute(self) -> 'Features':
        """
        :return: Returns a Features object where the feature map is precomputed on the feature data,
        i.e., some methods should be faster to evaluate on the precomputed Features object.
        """
        fm, fd = self.feature_map.precompute(self.feature_data)
        if self.diag is None:
            self.diag = fm.get_kernel_matrix_diag(fd)
        return Features(fm, fd, self.diag)

    def simplify(self) -> 'Features':
        """
        :return: Returns a Features object where the feature data is simplified (un-batched etc.),
        which potentially makes evaluations faster, similar to precompute().
        """
        return Features(self.feature_map, self.feature_data.simplify(), self.diag)

    def scale_tfm(self, factor: Optional[float] = None) -> 'FeaturesTransform':
        """
        :param factor: Factor by which to scale. If this is None,
        the factor is computed as the automatic scale normalization factor from the paper,
        on self.feature_data.
        :return: Returns a transformation object that scales a Features object by the given factor.
        """
        if factor is None:
            diag = self.get_kernel_matrix_diag()
            factor = 1.0 / math.sqrt(diag.mean().item())
        return LambdaFeaturesTransform(lambda f: Features(ScaledFeatureMap(f.feature_map, factor), f.feature_data))

    def posterior_tfm(self, sigma: float = 1.0, allow_kernel_space_posterior: bool = True, **config) \
            -> 'FeaturesTransform':
        """
        Computes the posterior transformation after observing self.feature_data.
        :param sigma: GP noise standard deviation.
        :param allow_kernel_space_posterior: Whether the method is allowed to use the kernel-space posterior formula
        if it is deemed more efficient but not strictly necessary. The kernel-space posterior feature map returned
        will not allow a computation of the feature matrix, which could be detrimental
        for methods that want to operate in feature space.
        :return: Returns a transformation object that replaces the feature map of a Features object
        by the posterior feature map arising from the GP with prior given by self.feature_map,
        after observing self.feature_data.
        """
        fm = self.feature_map.posterior(self.feature_data, sigma,
                                        allow_kernel_space_posterior=allow_kernel_space_posterior)
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, f.feature_data))

    def sketch_tfm(self, n_features: int, **config) -> 'FeaturesTransform':
        """
        Computes a sketching transformation for the current feature map.
        :param n_features: Number of target features of the sketched feature map.
        :param config: Optional parameters to specify sketching options.
        :return: Returns a Transformations that replaces the feature map of a Features object
        by the sketched version of self.feature_map.
        """
        fm = self.feature_map.sketch(n_features, **config)
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, f.feature_data))

    def acs_grad_tfm(self, sigma: float = 1.0) -> 'FeaturesTransform':
        """
        Computes the acs-grad transformation from the paper,
        using self.feature_data as training data to condition on.
        :param sigma: The GP noise standard deviation.
        :return: Returns a transformation object applying the acs-grad transformation.
        """
        post_features = self.posterior_tfm(sigma)(self)
        fm = ProductFeatureMap([self.feature_map, post_features.feature_map])
        if sigma != 1.0:
            fm = ScaledFeatureMap(fm, factor=1.0/sigma**4)
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, ListFeatureData([f.feature_data, f.feature_data])))

    def acs_rf_hyper_tfm(self, y_train: torch.Tensor, n_features: int, **config) \
            -> 'FeaturesTransform':
        """
        Computes the acs-rf-hyper transformation from the paper,
        using self.feature_data as training data to condition on.
        :param y_train: Labels corresponding to self.feature_data.
        Usually, self.feature_data corresponds to the training data, hence the name y_train.
        :param n_features: Number of target random features.
        :param config: Potential additional options (unused).
        :return: Returns a transformation that applies the acs-rf-hyper transformation to a Features object.
        """
        # adapted from https://github.com/rpinsler/active-bayesian-coresets
        x_train = self.get_feature_matrix()
        y_train = y_train.squeeze(1).type(x_train.type())
        in_features = x_train.shape[1]
        s = 1.0    # prior variance / length scale, I guess?
        # inverse gamma prior parameters
        a0 = 1.0
        b0 = 1.0
        y_var = b0 / a0
        w_cov_prior = s * torch.eye(in_features, device=x_train.device, dtype=x_train.dtype)
        xxw = x_train.t().matmul(x_train) + w_cov_prior
        L = robust_cholesky(xxw)  # xxw = L * L^T
        L_inv = torch.inverse(L)  # xxw^{-1} = L^{-T} * L^{-1}
        # theta_cov = torch.inverse(xxw)
        theta_cov = L_inv.t().matmul(L_inv)
        theta_mean = theta_cov.matmul(x_train.t().matmul(y_train))
        sigma_tilde_inv = xxw
        a_tilde = a0 + 0.5 * x_train.shape[0]
        b_tilde = b0 + 0.5 * ((y_train**2).sum() - torch.dot(theta_mean.t(), sigma_tilde_inv.matmul(theta_mean))).item()
        # theta_samples is be of shape in_features x n_features
        # if x \sim N(0, I), then L^{-T}x \sim N(0, L^{-T}L^{-1}) = N(0, theta_cov)
        theta_samples = L_inv.t().matmul(torch.randn(in_features, n_features, device=x_train.device, dtype=x_train.dtype))

        acs_rf_tfm = ACSRFHyperDataTransform(theta_cov=theta_cov, theta_samples=theta_samples, a_tilde=a_tilde,
                                             b_tilde=b_tilde, y_var=y_var)
        fm = SequentialFeatureMap(IdentityFeatureMap(n_features=n_features), [self.feature_map, acs_rf_tfm])
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, f.feature_data))

    def acs_rf_tfm(self, n_features: int, sigma: float = 1.0,
                      **config) -> 'FeaturesTransform':
        """
        Computes the acs-rf transformation from the paper,
        using self.feature_data as training data to condition on.
        :param n_features: Number of target random features.
        :param sigma: GP noise standard deviation.
        :param config: Potential additional options (unused).
        :return: Returns a transformation object applying the acs-rf transformation to a Features object.
        """
        # partially adapted from https://github.com/rpinsler/active-bayesian-coresets
        x_train = self.get_feature_matrix()
        in_features = x_train.shape[1]
        w_cov_prior = sigma**2 * torch.eye(in_features, device=x_train.device, dtype=x_train.dtype)
        xxw = x_train.t().matmul(x_train) + w_cov_prior
        L = robust_cholesky(xxw)  # xxw = L * L^T
        L_inv = torch.inverse(L)  # xxw^{-1} = L^{-T} * L^{-1}
        # theta_cov = torch.inverse(xxw)
        theta_cov = L_inv.t().matmul(L_inv)
        # theta_samples is be of shape in_features x n_features
        # if x \sim N(0, I), then L^{-T}x \sim N(0, L^{-T}L^{-1}) = N(0, theta_cov)
        theta_samples = L_inv.t().matmul(torch.randn(in_features, n_features, device=x_train.device, dtype=x_train.dtype))

        acs_rf_tfm = ACSRFDataTransform(theta_cov=theta_cov, theta_samples=theta_samples, sigma=sigma)
        fm = SequentialFeatureMap(IdentityFeatureMap(n_features=n_features), [self.feature_map, acs_rf_tfm])
        return LambdaFeaturesTransform(lambda f, fm=fm: Features(fm, f.feature_data))

    def get_n_samples(self) -> int:
        """
        :return: Returns the number of samples in self.feature_data.
        """
        return self.feature_data.get_n_samples()

    def __len__(self) -> int:
        """
        :return: Returns the number of samples in self.feature_data.
        """
        return self.get_n_samples()

    def get_n_features(self) -> int:
        """
        :return: Returns the number of features of the corresponding feature map.
        """
        return self.feature_map.get_n_features()

    def get_device(self) -> str:
        """
        :return: Returns the (torch) device that the feature data is on.
        """
        return self.feature_data.get_device()

    def get_dtype(self) -> Any:
        """
        :return: Returns the (torch) dtype that the feature data has.
        """
        return self.feature_data.get_dtype()

    def __getitem__(self, idxs: Union[int, slice, torch.Tensor]) -> 'Features':
        """
        Returns a Features object where the feature data is indexed by idxs. Note that if idxs is an integer,
        the feature data will be indexed by [idxs:idxs+1], i.e. the batch dimension will not be removed
        and the resulting feature data tensors will have shape [1, ...].
        This method is called when using an indexing expression
        such as features[0] or features[-2:] or features[torch.Tensor([0, 2, 4], dtype=torch.long)]
        :param idxs: Integer, slice, or torch.Tensor of integer type.
        See the comment above about indexing with integers.
        :return: Returns a Feature object where the feature data is the indexed version of self.feature_data.
        """
        idxs = Indexes(self.get_n_samples(), idxs)
        return Features(self.feature_map, self.feature_data[idxs],
                        None if self.diag is None else self.diag[idxs.get_idxs()])

    def get_kernel_matrix_diag(self) -> torch.Tensor:
        """
        Returns the kernel matrix diagonal
        obtained by self.feature_map.get_kernel_matrix_diag(self.feature_data).
        The kernel matrix diagonal is stored and reused,
        such that multiple calls to this method do not trigger multiple computations.
        Consequently, the returned Tensor should not be modified.
        :return: Returns a torch.Tensor of shape [len(self)] containing the kernel matrix diagonal.
        """
        if self.diag is None:
            self.diag = self.feature_map.get_kernel_matrix_diag(self.feature_data)
        return self.diag

    def get_kernel_matrix(self, other_features: 'Features') -> torch.Tensor:
        """
        Returns the kernel matrix k(self.feature_data, other_features.feature_data),
        where k is the kernel given by self.feature_map.
        :param other_features: Other features corresponding to the columns of the resulting kernel matrix.
        :return: Returns a torch.Tensor of shape [len(self), len(other_features)] corresponding to the kernel matrix.
        """
        return self.feature_map.get_kernel_matrix(self.feature_data, other_features.feature_data)

    def get_feature_matrix(self) -> torch.Tensor:
        """
        :return: Returns self.feature_map.get_feature_matrix(self.feature_data), i.e.,
        a torch.Tensor of shape [len(self), self.get_n_features()] containing the feature matrix.
        Note that this method cannot be used for all feature maps,
        since they might have an infinite-dimensional feature space.
        """
        return self.feature_map.get_feature_matrix(self.feature_data)

    def get_sq_dists(self, other_features: 'Features') -> torch.Tensor:
        """
        Return a matrix containing the squared feature space distances
        between self.feature_data and other_features.feature_data. These are computed using the kernel,
        hence they also work for feature maps with infinite-dimensional feature space.
        :param other_features: Features object to compute the distances to.
        :return: Returns a torch.Tensor of shape [len(self), len(other_features)]
        representing the squared distances between the feature data in feature space.
        """
        diag = self.get_kernel_matrix_diag()
        other_diag = other_features.get_kernel_matrix_diag()
        kernel_matrix = self.get_kernel_matrix(other_features)
        sq_dists = diag[:, None] + other_diag[None, :] - 2*kernel_matrix
        return sq_dists

    def batched(self, batch_size: int) -> 'Features':
        """
        Return a Features object that behaves as self,
        but where the feature data is virtually batched such that certain transformations are applied in batches.
        :param batch_size: Batch size of the batches. The last batch may be smaller.
        :return: Returns a Features object with batched feature data.
        """
        return Features(self.feature_map, self.feature_data.batched(batch_size), self.diag)

    def concat_with(self, other_features: 'Features'):
        """
        Concatenates two features objects along the sample dimension.
        :param other_features: Other Features object to concatenate with self.
        :return: Returns a Features object where self.feature_data and other_features.feature_data
        are concatenated along the batch dimension.
        """
        diag = torch.cat([self.diag, other_features.diag], dim=0) \
            if self.diag is not None and other_features.diag is not None \
            else None
        return Features(self.feature_map, ConcatFeatureData([self.feature_data, other_features.feature_data]), diag)


class ACSRFHyperDataTransform(DataTransform):
    """
    Internal helper class for the acs-rf-hyper transformation.
    This is essentially a feature map, transforming the data into random features.
    """
    def __init__(self, theta_cov: torch.Tensor, theta_samples: torch.Tensor, a_tilde: float, b_tilde: float,
                 y_var: float):
        super().__init__()
        self.theta_cov = theta_cov
        self.theta_samples = theta_samples
        self.a_tilde = a_tilde
        self.b_tilde = b_tilde
        self.y_var = y_var

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        x = feature_data.get_tensor(idxs)
        n_features = x.shape[1]
        pred_var = self.b_tilde / self.a_tilde * (1 + torch.sum(x @ self.theta_cov * x, dim=-1))
        const = -0.5 * math.log(2 * math.pi * self.y_var)
        # z = (x_pool @ theta_sample)[:, None]
        z = x.matmul(self.theta_samples)
        # expected_ll = const - 0.5 / y_var * (z ** 2 - 2 * pred_mean * z + pred_var + pred_mean ** 2)
        expected_ll = const - 0.5 / self.y_var * (z ** 2 + pred_var[:, None])
        return TensorFeatureData(math.sqrt(1 / n_features) * expected_ll)


class ACSRFDataTransform(DataTransform):
    """
    Internal helper class for the acs-rf transformation.
    This is essentially a feature map, transforming the data into random features.
    """
    def __init__(self, theta_cov: torch.Tensor, theta_samples: torch.Tensor, sigma: float):
        super().__init__()
        self.theta_cov = theta_cov
        self.theta_samples = theta_samples
        self.sigma = sigma

    def forward(self, feature_data: FeatureData, idxs: Indexes) -> FeatureData:
        x = feature_data.get_tensor(idxs)
        n_features = x.shape[1]
        k_train = torch.sum(x @ self.theta_cov * x, dim=-1)[:, None]
        const = 0.5 * torch.log(1 + k_train / self.sigma**2)
        # z = (x_pool @ theta_sample)[:, None]
        z = x.matmul(self.theta_samples)
        # expected_ll = const - 0.5 / y_var * (z ** 2 - 2 * pred_mean * z + pred_var + pred_mean ** 2)
        expected_ll = const - 0.5 / self.sigma**2 * (z ** 2 + k_train)
        return TensorFeatureData(math.sqrt(1 / n_features) * expected_ll)


class FeaturesTransform:
    """
    Abstract base class for classes that allow to transform a Features object into another Features object
    """
    def __call__(self, features: Features) -> Features:
        """
        This method should be overridden by subclasses.
        :param features: Features object to transform.
        :return: Returns the transformed Features object.
        """
        raise NotImplementedError()


class LambdaFeaturesTransform(FeaturesTransform):
    """
    FeaturesTransform subclass that simply applies a Callable object to the Features.
    """
    def __init__(self, f: Callable[[Features], Features]):
        """
        :param f: Function to apply to the Features object.
        """
        self.f = f

    def __call__(self, features: Features) -> Features:
        return self.f(features)


class SequentialFeaturesTransform(FeaturesTransform):
    """
    FeaturesTransform subclass that operates by applying multiple FeaturesTransform objects.
    """
    def __init__(self, tfms: List[FeaturesTransform]):
        self.tfms = tfms

    def __call__(self, features: Features) -> Features:
        for tfm in self.tfms:
            features = tfm(features)
        return features


class PrecomputeTransform(FeaturesTransform):
    """
    Transformation that precomputes Features, possibly with batching.
    """
    def __init__(self, batch_size: int = -1):
        """
        :param batch_size: Batch size to apply to the precomputation. Set to -1 if batching should not be used.
        """
        self.batch_size = batch_size

    def __call__(self, features: Features) -> Features:
        if self.batch_size > 0:
            features = features.batched(self.batch_size)
        return features.precompute().simplify()


class BatchTransform(FeaturesTransform):
    """
    Transformation that batches Features.
    """
    def __init__(self, batch_size: int):
        """
        :param batch_size: Batch size to apply to the Features object.
        """
        self.batch_size = batch_size

    def __call__(self, features: Features) -> Features:
        return features.batched(self.batch_size)


