import torch
import torch.nn as nn

from ..layers import *
from .features import *
from .selection import *


def select_batch(batch_size: int, models: List[nn.Module], data: Dict[str, FeatureData],
                 y_train: Optional[torch.Tensor],
                 base_kernel: str, kernel_transforms: List[Tuple[str, List]], selection_method: str,
                 precomp_batch_size: int = 32768, nn_batch_size=8192, **config) \
        -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    This method allows to apply the methods from the paper for selecting a batch of indices from the pool set,
    where the base kernel, kernel transformations and selection method can be easily configured.
    This method may reset the gradients of the parameters of provided models to None.
    :param batch_size: Number of samples to select from the pool set.
    :param models: List of NN models that are used for certain base kernels.
    Only one model is needed except if ensembling should be used.
    :param data: A dictionary such that data['train'] contains a FeatureData object representing the training data,
    and data['pool'] contains a FeatureData object representing the pool data.
    If the base kernels 'grad' or 'll' should be used, the get_tensor() method of the FeatureData objects
    should return tensors in a format that can be fed to the specified models.
    :param y_train: Tensor representing the training labels
    (which should be in the same order as the inputs in data['train']).
    This parameter is only used for the acs-rf-hyper transformation
    and can be set to None if this transformation is not used.
    :param base_kernel: Base kernel to use. Currently supported base kernels are
    'll', 'grad', 'lin', 'nngp', 'ntk', and 'laplace'.
    :param kernel_transforms: List of kernel transformations.
    Each kernel transformation is given by a tuple (name, args)
    and the corresponding transformation method is then called with parameters *args.
    The following kernel transformations are currently supported:
    ('train', [sigma]) applies k_{\to\Xtrain}
    ('train', [sigma, factor]) applies (factor^2 * k)_{\to post(\Xtrain, sigma^2)},
                                if factor is not None, or k_{\to\Xtrain} otherwise
    ('pool', [sigma]) does the same for \Xpool instead of \Xtrain
    ('pool', [sigma, factor]) does the same for \Xpool instead of \Xtrain
    ('scale', []) applies k_{\to scale(\Xtrain)}
    ('scale', [factor]) scales to factor^2 * k, if factor is not None, and k_{\to scale(\Xtrain)} otherwise
    ('rp', [n_features]) applies sketching (= random projections) with n_features random features
    ('ens', []) applies k_{\to ens}, i.e. ensembling of all kernels.
    ('acs-rf', [n_features, sigma]) applies k_{\to acs-rf(n_features)} with GP noise standard deviation sigma.
                                    As for 'train', there is also an optional third parameter
                                    for explicitly controlling the rescaling.
    ('acs-rf-hyper', [n_features]) applies k_{\to acs-rf-hyper(n_features)}.
                                    As for 'train', there is also an optional second parameter
                                    for explicitly controlling the rescaling.
    ('acs-grad', [sigma]) applies k_{\to acs-grad} with GP noise standard deviation sigma.
                                    As for 'train', there is also an optional second parameter
                                    for explicitly controlling the rescaling.
    :param selection_method: Selection method to apply. Currently supported options are
    'random', 'maxdiag', 'maxdet', 'bait', 'fw' (for FrankWolfe), 'maxdist', 'kmeanspp', 'lcmd'
    and the experimental options 'fw-kernel' (for FrankWolfe in kernel space, slow!), 'rmds' and 'sosd' (slow!)
    :param precomp_batch_size: Batch size used for precomputations of the features.
    :param nn_batch_size: Batch size used for passing the data through the NN.
    :param config: Other options. Examples:
    allow_float64=True: enables using float64 tensors if maxdet or transformations involving posteriors are used.
    compute_eff_dim=True: Triggers the computation of the effective dimension of the pool set kernel matrix
                            for kernels with feature space dimension <= 1000.
    sel_with_train=True/False: Forces TP/P-mode for the selection method.
                                By default, the distance-based methods run in TP-mode
                                and the other ones run in P-mode.
    allow_maxdet_fs=True: Allows to compute maxdet in feature space
                            if a criterion decides that it would be sensible in terms of efficiency.
    maxdet_sigma=<float>: sigma to use in maxdet, where sigma^2 is added to the kernel diagonal.
    allow_kernel_space_posterior=False: Do not compute the posterior in kernel space
                                        unless it is strictly necessary.
                                        This can be helpful if a feature-space representation is needed,
                                        for example for FrankWolfe.
    n_last_layers=<int> (default=1): How many of the last layers to use for the 'll' base kernel.
    layer_grad_dict=<Dict[Type, Type]> (default={nn.Linear: LinearGradientComputation}):
                            Allows to specify gradient computation classes for different types of layers
                            (subclasses of nn.Module). Note that layers that inherit from LayerGradientComputation
                            will be automatically used for gradient computation.
                            By the default value of this argument, nn.Linear layers also will be used for gradients.

    There are a few other options, e.g. for the nngp and ntk kernels,
    which can be found by searching for usages of 'config' in the source code.

    :return: Returns a tuple (batch_idxs, results) where batch_idxs is a torch.Tensor of shape [batch_size]
    containing the selected indices for the pool data. The dictionary results is of the form
    {'kernel_time': {'total': <float>, 'process': <float>},
     'selection_time': {'total': <float>, 'process': <float>},
     'selection_status': <None or status message>}
    and additionally may contain 'eff_dim': <float> if compute_eff_dim=True has been passed in **config.
    Times are measured in seconds.
    """
    bs = BatchSelectorImpl(models, data, y_train)
    return bs.select(base_kernel=base_kernel, kernel_transforms=kernel_transforms, selection_method=selection_method,
                     batch_size=batch_size, precomp_batch_size=precomp_batch_size, nn_batch_size=nn_batch_size,
                     **config)


class BatchSelectorImpl:
    """
    This internal class allows to apply the methods from the paper for selecting a batch from the pool set,
    where the base kernel, kernel transformations and selection method can be easily configured.
    A new object of this class should be created every time a batch is selected,
    since the state of the class gets modified during batch selection.
    """
    def __init__(self, models: List[nn.Module], data: Dict[str, FeatureData], y_train: Optional[torch.Tensor]):
        """
        :param models: List of trained NNs. Multiple NNs can be provided if ensembling should be used,
        otherwise only one NN should be provided.
        :param data: Dict of FeatureData objects, which should be in a format
        such that the NNs can be applied to the tensors contained in the FeatureData objects.
        The dict should at least contain the keys 'train' and 'pool', for training and pool data.
        Other keys can be provided and their data will also be transformed but they are currently not used,
        so they should not be provided.
        :param y_train: Labels on the training set as a torch.Tensor of shape [n_train, 1].
        This is only used for the acs-rf-hyper transform, otherwise it is sufficient to set y_train=None.
        """
        self.data = data
        self.models = models
        self.features = {}  # will be computed in select()
        self.n_models = len(models)
        self.y_train = y_train
        self.has_select_been_called = False
        self.device = self.data['train'].get_device()

    def apply_tfm(self, model_idx: int, tfm: FeaturesTransform):
        """
        Internal method that applies a transformation to all Features objects (train/pool)
        for the model with index model_idx.
        :param model_idx: Index of the model to apply transformations to
        :param tfm: Transformation to apply to the Features.
        """
        for key in self.features:
            self.features[key][model_idx] = tfm(self.features[key][model_idx])

    def ensemble(self):
        """
        Internal method to ensemble the kernels/features for different models.
        """
        for key in self.features:
            sum_fm = SumFeatureMap([f.feature_map for f in self.features[key]])
            sum_fd = ListFeatureData([f.feature_data for f in self.features[key]])
            self.features[key] = [Features(sum_fm, sum_fd)]
        self.n_models = 1

    def to_float64(self):
        """
        Internal method to convert the data to float64.
        This only has an effect on the result if it is applied before self.features is filled.
        """
        for key in self.data:
            self.data[key] = self.data[key].cast_to(torch.float64)

    def select(self, base_kernel: str, kernel_transforms: List[Tuple[str, List]], selection_method: str,
               batch_size: int, precomp_batch_size: int = 32768, nn_batch_size=8192, **config) \
            -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        This method allows to select a batch of indices from the pool set.
        The used Batch Active Learning method can be flexibly configured.
        This method may reset the gradients of the parameters of the model(s) provided in the constructor to None.
        :param base_kernel: Base kernel to use. Currently supported base kernels are
        'll', 'grad', 'lin', 'nngp', 'ntk', and 'laplace'.
        :param kernel_transforms: List of kernel transformations.
        Each kernel transformation is given by a tuple (name, args)
        and the corresponding transformation method is then called with parameters *args.
        The following kernel transformations are currently supported:
        ('train', [sigma]) applies k_{\to\Xtrain}
        ('train', [sigma, factor]) applies (factor^2 * k)_{\to post(\Xtrain, sigma^2)},
                                    if factor is not None, or k_{\to\Xtrain} otherwise
        ('pool', [sigma]) does the same for \Xpool instead of \Xtrain
        ('pool', [sigma, factor]) does the same for \Xpool instead of \Xtrain
        ('scale', []) applies k_{\to scale(\Xtrain)}
        ('scale', [factor]) scales to factor^2 * k, if factor is not None, and k_{\to scale(\Xtrain)} otherwise
        ('rp', [n_features]) applies sketching (= random projections) with n_features random features
        ('ens', []) applies k_{\to ens}, i.e. ensembling of all kernels.
        ('acs-rf', [n_features, sigma]) applies k_{\to acs-rf(n_features)} with GP noise standard deviation sigma.
                                        As for 'train', there is also an optional third parameter
                                        for explicitly controlling the rescaling.
        ('acs-rf-hyper', [n_features]) applies k_{\to acs-rf-hyper(n_features)}.
                                        As for 'train', there is also an optional second parameter
                                        for explicitly controlling the rescaling.
        ('acs-grad', [sigma]) applies k_{\to acs-grad} with GP noise standard deviation sigma.
                                        As for 'train', there is also an optional second parameter
                                        for explicitly controlling the rescaling.
        :param selection_method: Selection method to apply. Currently supported options are
        'random', 'maxdiag', 'maxdet', 'bait', 'fw' (for FrankWolfe), 'maxdist', 'kmeanspp', 'lcmd'
        and the experimental options 'fw-kernel' (for FrankWolfe in kernel space, slow!), 'rmds' and 'sosd' (slow!)
        :param batch_size: Number of samples to select from the pool set.
        :param precomp_batch_size: Batch size used for precomputations of the features.
        :param nn_batch_size: Batch size used for passing the data through the NN.
        :param config: Other options. Examples:
        allow_float64=True: enables using float64 tensors if maxdet or transformations involving posteriors are used.
        compute_eff_dim=True: Triggers the computation of the effective dimension of the pool set kernel matrix
                                for kernels with feature space dimension <= 1000.
        sel_with_train=True/False: Forces TP/P-mode for the selection method.
                                    By default, the distance-based methods run in TP-mode
                                    and the other ones run in P-mode.
        allow_maxdet_fs=True: Allows to compute maxdet in feature space
                                if a criterion decides that it would be sensible in terms of efficiency.
        maxdet_sigma=<float>: sigma to use in maxdet, where sigma^2 is added to the kernel diagonal.
        allow_kernel_space_posterior=False: Do not compute the posterior in kernel space
                                            unless it is strictly necessary.
                                            This can be helpful if a feature-space representation is needed,
                                            for example for FrankWolfe.
        n_last_layers=<int> (default=1): How many of the last layers to use for the 'll' base kernel.
        layer_grad_dict=<Dict[Type, Type]> (default={nn.Linear: LinearGradientComputation}):
                                Allows to specify gradient computation classes for different types of layers
                                (subclasses of nn.Module). Note that layers that inherit from LayerGradientComputation
                                will be automatically used for gradient computation.
                                By the default value of this argument, nn.Linear layers also will be used for gradients.
        verbosity=<int> (default=1): Allows to control how much information will be printed.
                                     Set to a value <= 0 if no information should be printed.
        use_cuda_synchronize=True: Use CUDA synchronize for more accurate time measurements.

        There are a few other options, e.g. for the nngp and ntk kernels,
        which can be found by searching for usages of 'config' in the source code.

        :return: Returns a tuple (batch_idxs, results) where batch_idxs is a torch.Tensor of shape [batch_size]
        containing the selected indices for the pool data. The dictionary results is of the form
        {'kernel_time': {'total': <float>, 'process': <float>},
         'selection_time': {'total': <float>, 'process': <float>},
         'selection_status': <None or status message>}
         and additionally may contain 'eff_dim': <float> if compute_eff_dim=True has been passed in **config.
        """

        if self.has_select_been_called:
            raise RuntimeError('select() can only be called once per BatchSelector object')
        self.has_select_been_called = True

        allow_tf32_before = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # do not use tf32 since it causes large numerical errors

        if config.get('allow_float64', False):
            use_float64 = (selection_method in ['maxdet', 'bait'])

            for tfm_name, tfm_args in kernel_transforms:
                if tfm_name in ['train', 'pool', 'acs-rf', 'acs-rf-hyper', 'acs-grad']:
                    use_float64 = True
        else:
            use_float64 = False

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        kernel_timer = utils.Timer()
        kernel_timer.start()

        if base_kernel == 'ntk':  # data -> features
            feature_maps = [ReLUNTKFeatureMap(n_layers=config.get('n_ntk_layers', len([
                l for l in model.modules() if isinstance(l, LayerGradientComputation)])),
                                              sigma_w_sq=config.get('weight_gain', 0.4)**2,
                                              sigma_b_sq=config.get('sigma_b', 0.0)**2) for model in self.models]
            if use_float64:
                self.to_float64()
        elif base_kernel == 'nngp':  # data -> features
            # we use sigma_b instead of bias_gain here, since in our case, biases are initialized to zero,
            # which for the nngp corresponds to sigma_b_sq = 0 instead of sigma_b_sq = bias_gain**2
            feature_maps = [ReLUNNGPFeatureMap(n_layers=config.get('n_nngp_layers', len([
                l for l in model.modules() if isinstance(l, LayerGradientComputation)])),
                                               sigma_w_sq=config.get('weight_gain', 0.25)**2,
                                               sigma_b_sq=config.get('sigma_b', 0.0)**2) for model in self.models]
            if use_float64:
                self.to_float64()
        elif base_kernel == 'grad':  # data -> features
            feature_maps = []
            grad_dict = config.get('layer_grad_dict', {nn.Linear: LinearGradientComputation})
            for model in self.models:
                grad_layers = []
                for layer in model.modules():
                    if isinstance(layer, LayerGradientComputation):
                        grad_layers.append(layer)
                    elif type(layer) in grad_dict:
                        grad_layers.append(grad_dict[type(layer)](layer))
                feature_maps.append(create_grad_feature_map(model, grad_layers, use_float64=use_float64))
        elif base_kernel == 'll':  # data -> features
            n_last_layers = config.get('n_last_layers', 1)
            feature_maps = []
            grad_dict = config.get('layer_grad_dict', {nn.Linear: LinearGradientComputation})
            for model in self.models:
                grad_layers = []
                for layer in model.modules():
                    if isinstance(layer, LayerGradientComputation):
                        grad_layers.append(layer)
                    elif type(layer) in grad_dict:
                        grad_layers.append(grad_dict[type(layer)](layer))
                feature_maps.append(create_grad_feature_map(model, grad_layers[-n_last_layers:],
                                                            use_float64=use_float64))
        elif base_kernel == 'linear':
            feature_maps = [IdentityFeatureMap(n_features=self.data['train'].get_tensor(0).shape[-1]) for model in self.models]
            if use_float64:
                self.to_float64()
        elif base_kernel == 'laplace':
            feature_maps = [LaplaceKernelFeatureMap(scale=config.get('laplace_scale', 1.0)) for model in self.models]
            if use_float64:
                self.to_float64()
        else:
            raise ValueError(f'Unknown base kernel "{base_kernel}"')

        self.features = {key: [Features(fm, feature_data) for fm in feature_maps]
                         for key, feature_data in self.data.items()}

        if base_kernel in ['ll', 'grad']:
            for i in range(self.n_models):
                # use smaller batch size for NN evaluation
                self.apply_tfm(i, BatchTransform(batch_size=nn_batch_size))

        for tfm_name, args in kernel_transforms:
            if tfm_name == 'train':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 2:
                        self.apply_tfm(i, self.features['train'][i].scale_tfm(factor=args[1]))
                    self.apply_tfm(i, self.features['train'][i].posterior_tfm(args[0], **config))
            elif tfm_name == 'pool':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 2:
                        self.apply_tfm(i, self.features['pool'][i].scale_tfm(factor=args[1]))
                    self.apply_tfm(i, self.features['pool'][i].posterior_tfm(args[0], **config))
            elif tfm_name == 'scale':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    self.apply_tfm(i, self.features['train'][i].scale_tfm(*args))
            elif tfm_name == 'rp':
                # don't precompute before random projections
                # since we might want to jointly forward through the model and project in batches
                # otherwise we might use more memory than needed
                for i in range(self.n_models):
                    self.apply_tfm(i, self.features['train'][i].sketch_tfm(*args, **config))
            elif tfm_name == 'ens':
                self.ensemble()
            elif tfm_name == 'acs-rf':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 3:
                        self.apply_tfm(i, self.features['train'][i].scale_tfm(factor=args[2]))
                    self.apply_tfm(i, self.features['train'][i].acs_rf_tfm(args[0], args[1]))
            elif tfm_name == 'acs-rf-hyper':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 2:
                        self.apply_tfm(i, self.features['train'][i].scale_tfm(factor=args[1]))
                    if self.y_train is None:
                        raise ValueError(
                            'Set y_train to None, but y_train is needed for the acs-rf-hyper transformation')
                    self.apply_tfm(i, self.features['train'][i].acs_rf_hyper_tfm(self.y_train, n_features=args[0]))
            elif tfm_name == 'acs-grad':
                for i in range(self.n_models):
                    self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))
                    if len(args) >= 2:
                        self.apply_tfm(i, self.features['train'][i].scale_tfm(factor=args[1]))
                    self.apply_tfm(i, self.features['train'][i].acs_grad_tfm(args[0]))
            else:
                raise ValueError(f'Unknown kernel transform "{tfm_name}"')

        for i in range(self.n_models):
            self.apply_tfm(i, PrecomputeTransform(batch_size=precomp_batch_size))

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        kernel_timer.pause()

        eff_dim = None
        if config.get('compute_eff_dim', False):
            pool_features = self.features['pool'][0]
            if pool_features.get_n_features() <= 1000:
                try:
                    feature_matrix = pool_features.get_feature_matrix()
                    feature_matrix = feature_matrix - feature_matrix.mean(dim=0, keepdim=True)
                    cov_matrix = feature_matrix.t().matmul(feature_matrix)
                    cov_matrix = 0.5 * (cov_matrix + cov_matrix.t())
                    largest_eigval = torch.linalg.matrix_norm(cov_matrix, 2)
                    eff_dim = (torch.trace(cov_matrix) / (torch.abs(largest_eigval) + 1e-30)).item()
                except Exception as e:
                    pass

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        selection_timer = utils.Timer()
        selection_timer.start()

        # only pick first model (if multiple models were there, they should have been ensembled by now)
        self.features = {key: val[0] for key, val in self.features.items()}

        if selection_method == 'random':
            alg = RandomSelectionMethod(self.features['pool'], **config)
        elif selection_method == 'maxdiag':
            alg = MaxDiagSelectionMethod(self.features['pool'], **config)
        elif selection_method == 'maxdet':
            sel_with_train = config.get('sel_with_train', None)
            n_select = batch_size
            n_features = self.features['pool'].get_n_features()
            maxdet_sigma = config.get('maxdet_sigma', 0.0)
            if sel_with_train:
                n_select += self.features['train'].get_n_samples()
            if config.get('allow_maxdet_fs', False) and n_features > 0 and n_features * 4 < n_select \
                    and maxdet_sigma > 0.0:
                alg = MaxDetFeatureSpaceSelectionMethod(self.features['pool'], self.features['train'],
                                                        noise_sigma=maxdet_sigma, **config)
            else:
                alg = MaxDetSelectionMethod(self.features['pool'], self.features['train'],
                                            noise_sigma=config.get('maxdet_sigma', 0.0), **config)
        elif selection_method == 'bait':
            alg = BaitFeatureSpaceSelectionMethod(self.features['pool'], self.features['train'],
                                            noise_sigma=config.get('bait_sigma', 0.0), **config)
        elif selection_method == 'maxdist':
            alg = MaxDistSelectionMethod(self.features['pool'], self.features['train'], **config)
        elif selection_method == 'lcmd':
            alg = LargestClusterMaxDistSelectionMethod(self.features['pool'], self.features['train'], **config)
        elif selection_method == 'rmds':
            alg = RandomizedMinDistSumSelectionMethod(self.features['pool'], self.features['train'],
                                                      max_n_candidates=config.get('max_rmds_candidates', 5), **config)
        elif selection_method == 'fw':
            alg = FrankWolfeSelectionMethod(self.features['pool'], self.features['train'], **config)
        elif selection_method == 'fw-kernel':
            # scales quadratically with pool set size
            alg = FrankWolfeKernelSpaceSelectionMethod(self.features['pool'], self.features['train'], **config)
        elif selection_method == 'kmeanspp':
            alg = KmeansppSelectionMethod(self.features['pool'], self.features['train'], **config)
        elif selection_method == 'sosd':
            # scales quadratically with pool set size
            alg = SumOfSquaredDistsSelectionMethod(self.features['pool'], self.features['train'], **config)
        else:
            raise ValueError(f'Unknown selection method "{selection_method}"')

        batch_idxs = alg.select(batch_size)

        if config.get('use_cuda_synchronize', False):
            torch.cuda.synchronize(self.device)

        selection_timer.pause()

        results_dict = {'kernel_time': kernel_timer.get_result_dict(),
                        'selection_time': selection_timer.get_result_dict(),
                        'selection_status': alg.get_status()}

        if eff_dim is not None:
            results_dict['eff_dim'] = eff_dim

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32_before

        return batch_idxs, results_dict
