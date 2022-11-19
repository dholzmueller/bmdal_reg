from .task_execution import *
from .train import ModelTrainer
from .data import *


# maybe some functionality for Runner to rerun if necessary but not print message that they are already exist if they do
# add more bait configurations


class RunConfigList:
    """
    Helper class holding a list of experiment configurations to be executed.
    """
    def __init__(self, configs: Optional[List[Tuple[float, ModelTrainer, float]]] = None):
        """
        Constructor.
        :param configs: Initialize with configs or empty list if configs is None.
        """
        self.configs = configs or []

    def append(self, ram_gb_per_sample: float, trainer: ModelTrainer, ram_gb_per_sample_bs: float = 0.0):
        """
        Append a configuration
        :param ram_gb_per_sample: RAM GB per sample constant used for RAM usage estimation of this configuration
        :param trainer: ModelTrainer that can be used to run BMDAL and contains the BMDAL configuration
        :param ram_gb_per_sample_bs: RAM GB per (sample * batch size) used for RAM usage estimation.
        This can be used for an additional term on top of ram_gb_per_sample, but is only helpful for some methods
        whose RAM usage increases significantly with the batch size.
        """
        self.configs.append((ram_gb_per_sample, trainer, ram_gb_per_sample_bs))

    def filter_names(self, alg_names: List[str]):
        """
        Filter out all configurations whose alg_names are not contained in alg_names.
        :param alg_names:
        :return:
        """
        return RunConfigList([(ram_gb_per_sample, trainer, ram_gb_per_sample_float)
                              for ram_gb_per_sample, trainer, ram_gb_per_sample_float in self.configs
                              if trainer.alg_name in alg_names])

    def __iter__(self) -> Iterable[Tuple[float, ModelTrainer, float]]:
        """
        Allows to iterate over the configs, e.g. within a for loop:
        for config in self:
            print(config)
        :return: Returns an iterator that outputs tuples (ram_gb_per_sample, trainer, ram_gb_per_sample_bs).
        """
        for cfg in self.configs:
            yield cfg


def get_bmdal_configs(**kwargs) -> RunConfigList:
    """
    :param kwargs: allows to set some hyperparameters, for example the learning rate, sigma_w, sigma_b, etc.
    :return: Returns a list of configurations for BMDAL used in the paper.
    """
    sigma = kwargs.get('post_sigma', 0.1)
    compute_eff_dim = True
    kwargs = utils.update_dict(dict(maxdet_sigma=sigma, bait_sigma=sigma, compute_eff_dim=compute_eff_dim,
                                    allow_float64=True, lr=0.375, weight_gain=0.2, bias_gain=0.2), kwargs)
    
    lst = RunConfigList()

    lst.append(1e-6, ModelTrainer(f'NN_random', selection_method='random',
                                         base_kernel='linear', kernel_transforms=[], **kwargs))

    # bait kernel comparison
    for fb_mode, overselection_factor in [('f', 1.0), ('fb', 2.0)]:
        lst.append(2e-5, ModelTrainer(f'NN_bait-{fb_mode}-p_ll_train', selection_method='bait',
                                      overselection_factor=overselection_factor, base_kernel='ll',
                                      kernel_transforms=[('train', [sigma, None])],
                                      **kwargs))
        lst.append(2e-5, ModelTrainer(f'NN_bait-{fb_mode}-p_grad_rp-512_train', selection_method='bait',
                                      overselection_factor=overselection_factor, base_kernel='grad',
                                      kernel_transforms=[('rp', [512]), ('train', [sigma, None])],
                                      **kwargs))
        if fb_mode == 'f':
            lst.append(2e-5, ModelTrainer(f'NN_bait-{fb_mode}-p_linear_train', selection_method='bait',
                                          overselection_factor=overselection_factor,
                                          base_kernel='linear', kernel_transforms=[('train', [sigma, None])],
                                          **kwargs))
            lst.append(2e-5, ModelTrainer(f'NN_bait-{fb_mode}-p_grad_ens-3_rp-512_train', selection_method='bait',
                                          overselection_factor=overselection_factor, base_kernel='grad', n_models=3,
                                          kernel_transforms=[('ens', []), ('rp', [512]),
                                                             ('train', [sigma, None])],
                                          **kwargs))
            lst.append(2e-5, ModelTrainer(f'NN_bait-{fb_mode}-p_ll_ens-3_rp-512_train', selection_method='bait',
                                          overselection_factor=overselection_factor, n_models=3,
                                          base_kernel='ll', kernel_transforms=[('ens', []),
                                                                               ('rp', [512]),
                                                                               ('train', [sigma, None])],
                                          **kwargs))

    # maxdet kernel comparison
    lst.append(2e-5, ModelTrainer(f'NN_maxdet-p_ll_train', selection_method='maxdet',
                                         base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs), 8e-9)
    lst.append(2.5e-5, ModelTrainer(f'NN_maxdet-p_grad_rp-512_train', selection_method='maxdet',
                                         base_kernel='grad',
                                         kernel_transforms=[('rp', [512]), ('train', [sigma, None])],
                                         **kwargs), 8e-9)
    lst.append(2.5e-5, ModelTrainer(f'NN_maxdet-p_grad_ens-3_rp-512_train', selection_method='maxdet',
                                         base_kernel='grad', n_models=3,
                                         kernel_transforms=[('ens', []), ('rp', [512]),
                                                            ('train', [sigma, None])],
                                         **kwargs), 8e-9)
    lst.append(2.5e-5,
               ModelTrainer(f'NN_maxdet-p_grad_rp-512_acs-rf-512', selection_method='maxdet',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf', [512, sigma, None])],
                            **kwargs), 8e-9)
    lst.append(2.5e-5,
               ModelTrainer(f'NN_maxdet-p_grad_rp-512_acs-grad', selection_method='maxdet',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-grad', [sigma, None])],
                            **kwargs), 8e-9)
    lst.append(2.5e-5,
               ModelTrainer(f'NN_maxdet-p_grad_rp-512_acs-rf-hyper-512', selection_method='maxdet',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf-hyper', [512, None])],
                            **kwargs), 8e-9)
    lst.append(2e-5,
               ModelTrainer(f'NN_maxdet-p_ll_acs-rf-512', selection_method='maxdet',
                            base_kernel='ll',
                            kernel_transforms=[('acs-rf', [512, sigma, None])],
                            **kwargs), 8e-9)

    lst.append(8e-5, ModelTrainer(f'NN_maxdet-tp_grad_scale', selection_method='maxdet',
                                         base_kernel='grad', sel_with_train=True,
                                         kernel_transforms=[('scale', [None])],
                                         **kwargs), 8e-9)
    lst.append(2e-5, ModelTrainer(f'NN_maxdet-p_ll_ens-3_rp-512_train', selection_method='maxdet',
                                         n_models=3,
                                         base_kernel='ll', kernel_transforms=[('ens', []),
                                                                              ('rp', [512]),
                                                                              ('train', [sigma, None])],
                                         **kwargs), 8e-9)
    lst.append(8e-5, ModelTrainer(f'NN_maxdet-tp_nngp_scale', selection_method='maxdet',
                                         base_kernel='nngp', sel_with_train=True,
                                         kernel_transforms=[('scale', [None])],
                                         **kwargs), 8e-9)
    lst.append(2e-5, ModelTrainer(f'NN_maxdet-p_linear_train', selection_method='maxdet',
                                         base_kernel='linear', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs), 8e-9)

    # maxdist, kmeanspp, lcmd kernel comparisons
    for sel_name in ['maxdist', 'kmeanspp', 'lcmd']:
        lst.append(2e-6, ModelTrainer(f'NN_{sel_name}-tp_linear', selection_method=sel_name,
                                             base_kernel='linear', kernel_transforms=[], **kwargs))
        lst.append(2e-6, ModelTrainer(f'NN_{sel_name}-tp_nngp', selection_method=sel_name,
                                             base_kernel='nngp', kernel_transforms=[], **kwargs))
        lst.append(4e-6, ModelTrainer(f'NN_{sel_name}-tp_ll', selection_method=sel_name,
                                             base_kernel='ll', kernel_transforms=[], **kwargs))
        lst.append(2e-5, ModelTrainer(f'NN_{sel_name}-tp_grad', selection_method=sel_name,
                                             base_kernel='grad', kernel_transforms=[], **kwargs))
        lst.append(4e-6, ModelTrainer(f'NN_{sel_name}-tp_grad_rp-512', selection_method=sel_name,
                                             base_kernel='grad', kernel_transforms=[('rp', [512])], **kwargs))
        lst.append(8e-6, ModelTrainer(f'NN_{sel_name}-tp_grad_ens-3_rp-512', selection_method=sel_name,
                                             n_models=3,
                                             base_kernel='grad', kernel_transforms=[('ens', []), ('rp', [512])],
                                             **kwargs))
        lst.append(2e-5, ModelTrainer(f'NN_{sel_name}-p_ll_train', selection_method=sel_name,
                                             sel_with_train=False,
                                             base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                             **kwargs))
        lst.append(2e-5, ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_train', selection_method=sel_name,
                                             sel_with_train=False,
                                             base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                    ('train', [sigma, None])],
                                             **kwargs))
        lst.append(1e-5, ModelTrainer(f'NN_{sel_name}-tp_ll_ens-3_rp-512', selection_method=sel_name,
                                             n_models=3,
                                             base_kernel='ll', kernel_transforms=[('ens', []), ('rp', [512])],
                                             **kwargs))
        lst.append(2e-5,
                   ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_acs-rf-512', selection_method=sel_name,
                                sel_with_train=False,
                                base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                       ('acs-rf', [512, sigma, None])],
                                **kwargs))
        lst.append(2e-5,
                   ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_acs-grad', selection_method=sel_name,
                                sel_with_train=False,
                                base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                       ('acs-grad', [sigma, None])],
                                **kwargs))
        lst.append(2e-5,
                   ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_acs-rf-hyper-512', selection_method=sel_name,
                                sel_with_train=False,
                                base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                       ('acs-rf-hyper', [512, None])],
                                **kwargs))
        lst.append(2e-5, ModelTrainer(f'NN_{sel_name}-p_ll_acs-rf-512', selection_method=sel_name,
                                             sel_with_train=False,
                                             base_kernel='ll',
                                             kernel_transforms=[('acs-rf', [512, sigma, None])],
                                             **kwargs))

    # maxdiag kernel comparison
    lst.append(8e-6, ModelTrainer(f'NN_maxdiag_ll_train', selection_method='maxdiag',
                                         base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    lst.append(8e-6, ModelTrainer(f'NN_maxdiag_grad_rp-512_train', selection_method='maxdiag',
                                         base_kernel='grad',
                                         kernel_transforms=[('rp', [512]), ('train', [sigma, None])],
                                         **kwargs))
    lst.append(8e-6, ModelTrainer(f'NN_maxdiag_grad_ens-3_rp-512_train', selection_method='maxdiag',
                                         base_kernel='grad', n_models=3,
                                         kernel_transforms=[('ens', []), ('rp', [512]),
                                                            ('train', [sigma, None])],
                                         **kwargs))
    lst.append(1.3e-5,
               ModelTrainer(f'NN_maxdiag_grad_rp-512_acs-rf-512', selection_method='maxdiag',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf', [512, sigma, None])],
                            **kwargs))
    lst.append(1.3e-5,
               ModelTrainer(f'NN_maxdiag_grad_rp-512_acs-grad', selection_method='maxdiag',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-grad', [sigma, None])],
                            **kwargs))
    lst.append(1.3e-5,
               ModelTrainer(f'NN_maxdiag_grad_rp-512_acs-rf-hyper-512', selection_method='maxdiag',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf-hyper', [512, None])],
                            **kwargs))
    lst.append(1.3e-5,
               ModelTrainer(f'NN_maxdiag_ll_acs-rf-512', selection_method='maxdiag',
                            base_kernel='ll',
                            kernel_transforms=[('acs-rf', [512, sigma, None])]))
    lst.append(8e-6, ModelTrainer(f'NN_maxdiag_linear_train', selection_method='maxdiag',
                                         base_kernel='linear', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    lst.append(8e-5, ModelTrainer(f'NN_maxdiag_nngp_train', selection_method='maxdiag',
                                         base_kernel='nngp', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    lst.append(8e-5, ModelTrainer(f'NN_maxdiag_grad_train', selection_method='maxdiag',
                                         base_kernel='grad', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))

    # Frank-Wolfe kernel comparison
    lst.append(2e-5, ModelTrainer(f'NN_fw-p_ll_train', selection_method='fw',
                                         base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    lst.append(2e-5, ModelTrainer(f'NN_fw-p_grad_rp-512_train', selection_method='fw',
                                         base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                ('train', [sigma, None])],
                                         **kwargs))
    lst.append(2e-5, ModelTrainer(f'NN_fw-p_ll_acs-grad_rp-512', selection_method='fw',
                                         base_kernel='ll', kernel_transforms=[('acs-grad', [sigma, None]),
                                                                              ('rp', [512])],
                                         **kwargs))
    lst.append(2e-5, ModelTrainer(f'NN_fw-p_grad_rp-512_acs-grad_rp-512', selection_method='fw',
                                         base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                ('acs-grad', [sigma, None]),
                                                                                ('rp', [512])],
                                         **kwargs))
    lst.append(2e-5, ModelTrainer(f'NN_fw-p_ll_acs-rf-512', selection_method='fw',
                                         base_kernel='ll', kernel_transforms=[('acs-rf', [512, sigma, None])],
                                         **kwargs))

    lst.append(2e-5, ModelTrainer(f'NN_fw-p_grad_rp-512_acs-rf-512', selection_method='fw',
                                         base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                ('acs-rf', [512, sigma, None])],
                                         **kwargs))
    lst.append(2e-5, ModelTrainer(f'NN_fw-p_ll_acs-rf-hyper-512', selection_method='fw',
                                         base_kernel='ll',
                                         kernel_transforms=[('acs-rf-hyper', [512, None])],
                                         **kwargs))
    lst.append(2e-5,
               ModelTrainer(f'NN_fw-p_grad_rp-512_acs-rf-hyper-512', selection_method='fw',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf-hyper', [512, None])],
                            **kwargs))

    return lst


def get_relu_tuning_configs() -> RunConfigList:
    lst = RunConfigList()

    for lr in [3e-2, 5e-2, 8e-2]:
        for sigma_w in [0.25, 0.4, 0.7, 1.0, 1.414]:
            for wd in [1e-2, 1e-3, 0.0]:
                lst.append(1e-6, ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}',
                                                     lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                                     selection_method='random', base_kernel='linear',
                                                     kernel_transforms=[]))
    for lr in [8e-2, 1e-1, 2e-1]:
        for sigma_w in [0.25, 0.4, 0.5]:
            for wd in [1e-2, 1e-3, 1e-1, 0.0, 3e-3]:
                lst.append(1e-6, ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}',
                                                     lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                                     selection_method='random', base_kernel='linear',
                                                     kernel_transforms=[]))

    for lr in [2e-1, 3e-1, 4e-1]:
        for sigma_w in [0.25]:
            for wd in [0.0, 1e-3, 3e-3]:
                lst.append(1e-6, ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}',
                                                     lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                                     selection_method='random', base_kernel='linear',
                                                     kernel_transforms=[]))

    for lr in [7.5e-2]:
        for sigma_w in [1.0]:
            for wd in [0.0]:
                for wig in [0.25]:
                    for sigma_b in [0.1, 0.4, 1.0]:
                        lst.append(1e-6,
                                   ModelTrainer(
                                       f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}_wig-{wig:g}-sigmab-{sigma_b:g}',
                                       lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                       weight_init_gain=wig, bias_gain=sigma_b,
                                       selection_method='random', base_kernel='linear',
                                       kernel_transforms=[]))

    for lr in [7.5e-2]:
        for sigma_w in [1.0]:
            for wd in [0.0]:
                for wig in [0.1, 1.0]:
                    for sigma_b in [1.0]:
                        lst.append(1e-6,
                                   ModelTrainer(
                                       f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}_wig-{wig:g}-sigmab-{sigma_b:g}',
                                       lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                       weight_init_gain=wig, bias_gain=sigma_b,
                                       selection_method='random', base_kernel='linear',
                                       kernel_transforms=[]))

    for lr in [7.5e-2, 1e-1]:
        for sigma_w in [1.0]:
            for wd in [0.0]:
                for wig in [0.1, 0.25, 0.5]:
                    for sigma_b in [1.0]:
                        lst.append(1e-6,
                                   ModelTrainer(
                                       f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}_wig-{wig:g}-sigmab-{sigma_b:g}',
                                       lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                       weight_init_gain=wig, bias_gain=sigma_b,
                                       selection_method='random', base_kernel='linear',
                                       kernel_transforms=[]))

    for sigma_w in [0.1, 0.2, 0.25, 0.3]:
        for sigma_b in [0.1, 0.2, 0.3]:
            for lr in [5e-2 / sigma_w, 7.5e-2 / sigma_w, 1e-1 / sigma_w]:
                wd = 0.0
                wig = 1.0
                lst.append(1e-6,
                           ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}_wig-{wig:g}-sigmab-{sigma_b:g}',
                                        lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                        weight_init_gain=wig, bias_gain=sigma_b,
                                        selection_method='random', base_kernel='linear',
                                        kernel_transforms=[]))

    sigma_w = 0.2
    sigma_b = 0.2
    for lr in [5e-2 / sigma_w, 7.5e-2 / sigma_w, 1e-1 / sigma_w]:
        for lr_sched in ['hat', 'warmup']:
            wd = 0.0
            wig = 1.0
            lst.append(1e-6,
                       ModelTrainer(
                           f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}_wig-{wig:g}-sigmab-{sigma_b:g}-{lr_sched}',
                           lr=lr, weight_decay=wd, weight_gain=sigma_w,
                           weight_init_gain=wig, bias_gain=sigma_b, lr_sched=lr_sched,
                           selection_method='random', base_kernel='linear',
                           kernel_transforms=[]))

    return lst


def get_silu_tuning_configs() -> RunConfigList:
    lst = RunConfigList()

    for sigma_w in [0.2, 0.5, 1.0]:
        for sigma_b in [0.1, 0.25, 0.5, 1.0]:
            for lr in [3e-2 / sigma_w, 5e-2 / sigma_w, 7.5e-2 / sigma_w, 1e-1 / sigma_w]:
                wd = 0.0
                wig = 1.0
                lst.append(1e-6,
                           ModelTrainer(f'NN_sigmaw-{sigma_w:g}_wd-{wd:g}_sigmab-{sigma_b:g}_lr-{lr:g}',
                                        lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                        weight_init_gain=wig, bias_gain=sigma_b, act='silu',
                                        selection_method='random', base_kernel='linear',
                                        kernel_transforms=[]))

    for wd in [1e-2, 1e-3]:
        sigma_w = 0.5
        sigma_b = 1.0
        lr = 0.15
        wig = 1.0
        lst.append(1e-6,
                   ModelTrainer(f'NN_sigmaw-{sigma_w:g}_wd-{wd:g}_sigmab-{sigma_b:g}_lr-{lr:g}',
                                lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                weight_init_gain=wig, bias_gain=sigma_b, act='silu',
                                selection_method='random', base_kernel='linear',
                                kernel_transforms=[]))

    return lst


def run_experiments(exp_name: str, n_splits: int, run_config_list: RunConfigList,
                    batch_sizes_configs: Optional[List[List[int]]] = None, task_descs: Optional[List[str]] = None,
                    use_pool_for_normalization: bool = True, max_jobs_per_device: int = 20,
                    n_train_initial: int = 256, ds_names: Optional[List[str]] = None,
                    sequential_split: Optional[int] = 9):
    """
    This function allows to run experiments in a parallelized fashion.
    :param exp_name: Name for the group of experiments. This name will be used as a folder name.
    :param run_config_list: List of configurations that should be run.
    :param n_splits: Number of random splits to run.
    :param batch_sizes_configs: Optional list of lists of batch sizes.
    The callback function will be called once for each list of batch sizes,
    with tasks using this list of batch sizes for BMDAL.
    By default, batch_sizes_configs=[[256]*16] will be used as in the paper.
    :param task_descs: Optional list of task descriptions, which will be appended to the dataset names.
    One task description per list of batch sizes in batch_sizes_configs should be provided.
    :param use_pool_for_normalization: If True, compute the statistics for standardizing the inputs of the data sets
    based on the initial training and pool set. Otherwise, compute them only on the initial training set.
    :param max_jobs_per_device: Maximum number of processes run per device.
    If GPUs are available, each GPU is one device. Otherwise, the CPU is used as a single device.
    :param n_train_initial: Initial training set size. Defaults to 256 as in the paper.
    :param ds_names: Names of data sets that should be used. By default, all data sets from the benchmark are used.
    :param sequential_split: ID of the random split where max_jobs_per_device is set to 1
    for accurate timing statistics. Defaults to 9. If no split should be used for timing, set this to None.
    """
    if ds_names is None:
        ds_names = ['sgemm', 'mlr_knn_rng', 'wecs', 'ct', 'kegg_undir_uci', 'online_video', 'query_agg_count',
                    'poker', 'road_network', 'fried', 'diamonds', 'methane', 'protein', 'sarcos', 'stock']

    if batch_sizes_configs is None:
        batch_sizes_configs = [[256] * 16]
    if task_descs is None:
        task_descs = ['256x16']

    tabular_tasks = Task.get_tabular_tasks(n_train=n_train_initial, al_batch_sizes=[], ds_names=ds_names)

    for t in tabular_tasks:
        print(f'Task {t.task_name} has n_pool={t.n_pool}, n_test={t.n_test}, n_features={t.data_info.n_features}')

    for max_split_id in range(0, n_splits):
        # run each split sequentially
        print(f'Running all configurations on split {max_split_id}')
        # run only one experiment per GPU on split sequential_split for timing experiments
        do_timing = max_split_id == sequential_split
        scheduler = JobScheduler(max_jobs_per_device=1 if do_timing else max_jobs_per_device)
        runner = JobRunner(scheduler=scheduler)
        for split_id in range(0, max_split_id+1):
            for batch_sizes_config, task_desc in zip(batch_sizes_configs, task_descs):
                tasks = Task.get_tabular_tasks(n_train=n_train_initial, al_batch_sizes=batch_sizes_config,
                                               ds_names=ds_names,
                                               desc=task_desc)
                for ram_gb_per_sample, trainer, ram_gb_per_sample_bs in run_config_list:
                    runner.add(exp_name, split_id, tasks, ram_gb_per_sample, trainer, do_timing=do_timing,
                               warn_if_exists=(split_id == max_split_id),
                               use_pool_for_normalization=use_pool_for_normalization,
                               ram_gb_per_sample_bs=ram_gb_per_sample_bs)
        runner.run_all()


def get_relu_configs() -> RunConfigList:
    return get_bmdal_configs(weight_gain=0.2, bias_gain=0.2, post_sigma=1e-3, lr=0.375, act='relu')


def get_silu_configs() -> RunConfigList:
    return get_bmdal_configs(weight_gain=0.5, bias_gain=1.0, post_sigma=1e-3, lr=0.15, act='silu')


if __name__ == '__main__':
    use_pool_for_normalization = True

    relu_bs_configs = get_relu_configs().filter_names(
        ['NN_lcmd-tp_grad_rp-512', 'NN_kmeanspp-p_grad_rp-512_acs-rf-512', 'NN_fw-p_grad_rp-512_acs-rf-hyper-512',
         'NN_maxdist-p_grad_rp-512_train',
         'NN_maxdet-p_grad_rp-512_train',
         'NN_maxdiag_grad_rp-512_acs-rf-512',
         'NN_bait-f-p_grad_rp-512_train'])

    # # ReLU experiments
    run_experiments('relu', 20, get_relu_configs(),
                    use_pool_for_normalization=use_pool_for_normalization)
    # SiLU experiments, without batch size experiments
    run_experiments('silu', 20, get_silu_configs(),
                    use_pool_for_normalization=use_pool_for_normalization)
    # # ReLU batch size experiments
    run_experiments('relu', 20, relu_bs_configs,
                    batch_sizes_configs=[[2**(12-m)]*(2**m) for m in range(7) if m != 4],
                    task_descs=[f'{2**(12-m)}x{2**m}' for m in range(7) if m != 4],
                    use_pool_for_normalization=use_pool_for_normalization)

    # for hyperparameter optimization
    # run_experiments('relu_tuning', 2, get_relu_tuning_configs(),
    #                 use_pool_for_normalization=use_pool_for_normalization)
    # run_experiments('silu_tuning', 2, get_silu_tuning_configs(),
    #                 use_pool_for_normalization=use_pool_for_normalization)
