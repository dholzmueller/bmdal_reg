from task_execution import *
from train import ModelTrainer
from data import *


def add_all_configs_to_runner(runner: JobRunner, tasks: List[Task], **kwargs):
    # function that adds experiment configurations to a runner which can then execute them later
    sigma = kwargs.get('post_sigma', 0.1)
    compute_eff_dim = True
    kwargs = utils.update_dict(dict(maxdet_sigma=sigma, compute_eff_dim=compute_eff_dim,
                                    lr=0.375, weight_gain=0.2, bias_gain=0.2), kwargs)

    runner.add(tasks, 1e-6, ModelTrainer(f'NN_random', selection_method='random',
                                         base_kernel='linear', kernel_transforms=[], **kwargs))

    for sel_name in ['maxdist', 'kmeanspp', 'lcmd']:
        runner.add(tasks, 2e-6, ModelTrainer(f'NN_{sel_name}-tp_linear', selection_method=sel_name,
                                             base_kernel='linear', kernel_transforms=[], **kwargs))
        runner.add(tasks, 2e-6, ModelTrainer(f'NN_{sel_name}-tp_nngp', selection_method=sel_name,
                                             base_kernel='nngp', kernel_transforms=[], **kwargs))
        runner.add(tasks, 4e-6, ModelTrainer(f'NN_{sel_name}-tp_ll', selection_method=sel_name,
                                             base_kernel='ll', kernel_transforms=[], **kwargs))
        runner.add(tasks, 2e-5, ModelTrainer(f'NN_{sel_name}-tp_grad', selection_method=sel_name,
                                             base_kernel='grad', kernel_transforms=[], **kwargs))
        runner.add(tasks, 4e-6, ModelTrainer(f'NN_{sel_name}-tp_grad_rp-512', selection_method=sel_name,
                                             base_kernel='grad', kernel_transforms=[('rp', [512])], **kwargs))
        runner.add(tasks, 8e-6, ModelTrainer(f'NN_{sel_name}-tp_grad_ens-3_rp-512', selection_method=sel_name,
                                             n_models=3,
                                             base_kernel='grad', kernel_transforms=[('ens', []), ('rp', [512])],
                                             **kwargs))
        runner.add(tasks, 6e-6, ModelTrainer(f'NN_{sel_name}-p_ll_train', selection_method=sel_name,
                                             sel_with_train=False,
                                             base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                             **kwargs))
        runner.add(tasks, 6e-6, ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_train', selection_method=sel_name,
                                             sel_with_train=False,
                                             base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                    ('train', [sigma, None])],
                                             **kwargs))
        runner.add(tasks, 8e-6, ModelTrainer(f'NN_{sel_name}-tp_ll_ens-3_rp-512', selection_method=sel_name,
                                             n_models=3,
                                             base_kernel='ll', kernel_transforms=[('ens', []), ('rp', [512])],
                                             **kwargs))
        runner.add(tasks, 8e-6,
                   ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_acs-rf-512', selection_method=sel_name,
                                sel_with_train=False,
                                base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                       ('acs-rf', [512, sigma, None])],
                                **kwargs))
        runner.add(tasks, 8e-6,
                   ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_acs-grad', selection_method=sel_name,
                                sel_with_train=False,
                                base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                       ('acs-grad', [sigma, None])],
                                **kwargs))
        runner.add(tasks, 8e-6,
                   ModelTrainer(f'NN_{sel_name}-p_grad_rp-512_acs-rf-hyper-512', selection_method=sel_name,
                                sel_with_train=False,
                                base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                       ('acs-rf-hyper', [512, None])],
                                **kwargs))
        runner.add(tasks, 8e-6, ModelTrainer(f'NN_{sel_name}-p_ll_acs-rf-512', selection_method=sel_name,
                                             sel_with_train=False,
                                             base_kernel='ll',
                                             kernel_transforms=[('acs-rf', [512, sigma, None])],
                                             **kwargs))

    # maxdet kernel comparison
    runner.add(tasks, 2e-5, ModelTrainer(f'NN_maxdet-p_ll_train', selection_method='maxdet',
                                         base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 2e-5, ModelTrainer(f'NN_maxdet-p_grad_rp-512_train', selection_method='maxdet',
                                         base_kernel='grad',
                                         kernel_transforms=[('rp', [512]), ('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 2e-5, ModelTrainer(f'NN_maxdet-p_grad_ens-3_rp-512_train', selection_method='maxdet',
                                         base_kernel='grad', n_models=3,
                                         kernel_transforms=[('ens', []), ('rp', [512]),
                                                            ('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 2e-5,
               ModelTrainer(f'NN_maxdet-p_grad_rp-512_acs-rf-512', selection_method='maxdet',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf', [512, sigma, None])],
                            **kwargs))
    runner.add(tasks, 2e-5,
               ModelTrainer(f'NN_maxdet-p_grad_rp-512_acs-grad', selection_method='maxdet',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-grad', [sigma, None])],
                            **kwargs))
    runner.add(tasks, 2e-5,
               ModelTrainer(f'NN_maxdet-p_grad_rp-512_acs-rf-hyper-512', selection_method='maxdet',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf-hyper', [512, None])],
                            **kwargs))
    runner.add(tasks, 2e-5,
               ModelTrainer(f'NN_maxdet-p_ll_acs-rf-512', selection_method='maxdet',
                            base_kernel='ll',
                            kernel_transforms=[('acs-rf', [512, sigma, None])],
                            **kwargs))

    runner.add(tasks, 7e-5, ModelTrainer(f'NN_maxdet-tp_grad_scale', selection_method='maxdet',
                                         base_kernel='grad', sel_with_train=True,
                                         kernel_transforms=[('scale', [None])],
                                         **kwargs))
    runner.add(tasks, 2e-5, ModelTrainer(f'NN_maxdet-p_ll_ens-3_rp-512_train', selection_method='maxdet',
                                         n_models=3,
                                         base_kernel='ll', kernel_transforms=[('ens', []),
                                                                              ('rp', [512]),
                                                                              ('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 7e-5, ModelTrainer(f'NN_maxdet-tp_nngp_scale', selection_method='maxdet',
                                         base_kernel='nngp', sel_with_train=True,
                                         kernel_transforms=[('scale', [None])],
                                         **kwargs))
    runner.add(tasks, 2e-5, ModelTrainer(f'NN_maxdet-p_linear_train', selection_method='maxdet',
                                         base_kernel='linear', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))

    # maxdiag kernel comparison
    runner.add(tasks, 5e-6, ModelTrainer(f'NN_maxdiag_ll_train', selection_method='maxdiag',
                                         base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 5e-6, ModelTrainer(f'NN_maxdiag_grad_rp-512_train', selection_method='maxdiag',
                                         base_kernel='grad',
                                         kernel_transforms=[('rp', [512]), ('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 5e-6, ModelTrainer(f'NN_maxdiag_grad_ens-3_rp-512_train', selection_method='maxdiag',
                                         base_kernel='grad', n_models=3,
                                         kernel_transforms=[('ens', []), ('rp', [512]),
                                                            ('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 5e-6,
               ModelTrainer(f'NN_maxdiag_grad_rp-512_acs-rf-512', selection_method='maxdiag',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf', [512, sigma, None])],
                            **kwargs))
    runner.add(tasks, 5e-6,
               ModelTrainer(f'NN_maxdiag_grad_rp-512_acs-grad', selection_method='maxdiag',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-grad', [sigma, None])],
                            **kwargs))
    runner.add(tasks, 5e-6,
               ModelTrainer(f'NN_maxdiag_grad_rp-512_acs-rf-hyper-512', selection_method='maxdiag',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf-hyper', [512, None])],
                            **kwargs))
    runner.add(tasks, 5e-6,
               ModelTrainer(f'NN_maxdiag_ll_acs-rf-512', selection_method='maxdiag',
                            base_kernel='ll',
                            kernel_transforms=[('acs-rf', [512, sigma, None])]))
    runner.add(tasks, 5e-6, ModelTrainer(f'NN_maxdiag_linear_train', selection_method='maxdiag',
                                         base_kernel='linear', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 7e-5, ModelTrainer(f'NN_maxdiag_nngp_train', selection_method='maxdiag',
                                         base_kernel='nngp', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 7e-5, ModelTrainer(f'NN_maxdiag_grad_train', selection_method='maxdiag',
                                         base_kernel='grad', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))

    # frank-wolfe kernel comparison
    runner.add(tasks, 6e-6, ModelTrainer(f'NN_fw-p_ll_train', selection_method='fw',
                                         base_kernel='ll', kernel_transforms=[('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 6e-6, ModelTrainer(f'NN_fw-p_grad_rp-512_train', selection_method='fw',
                                         base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                ('train', [sigma, None])],
                                         **kwargs))
    runner.add(tasks, 8e-6, ModelTrainer(f'NN_fw-p_ll_acs-grad_rp-512', selection_method='fw',
                                         base_kernel='ll', kernel_transforms=[('acs-grad', [sigma, None]),
                                                                              ('rp', [512])],
                                         **kwargs))
    runner.add(tasks, 8e-6, ModelTrainer(f'NN_fw-p_grad_rp-512_acs-grad_rp-512', selection_method='fw',
                                         base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                ('acs-grad', [sigma, None]),
                                                                                ('rp', [512])],
                                         **kwargs))
    runner.add(tasks, 8e-6, ModelTrainer(f'NN_fw-p_ll_acs-rf-512', selection_method='fw',
                                         base_kernel='ll', kernel_transforms=[('acs-rf', [512, sigma, None])],
                                         **kwargs))

    runner.add(tasks, 8e-6, ModelTrainer(f'NN_fw-p_grad_rp-512_acs-rf-512', selection_method='fw',
                                         base_kernel='grad', kernel_transforms=[('rp', [512]),
                                                                                ('acs-rf', [512, sigma, None])],
                                         **kwargs))
    runner.add(tasks, 8e-6, ModelTrainer(f'NN_fw-p_ll_acs-rf-hyper-512', selection_method='fw',
                                         base_kernel='ll',
                                         kernel_transforms=[('acs-rf-hyper', [512, None])],
                                         **kwargs))
    runner.add(tasks, 8e-6,
               ModelTrainer(f'NN_fw-p_grad_rp-512_acs-rf-hyper-512', selection_method='fw',
                            base_kernel='grad',
                            kernel_transforms=[('rp', [512]), ('acs-rf-hyper', [512, None])],
                            **kwargs))


def add_relu_tuning_configs_to_runner(runner: JobRunner, tasks: List[Task]):
    for lr in [3e-2, 5e-2, 8e-2]:
        for sigma_w in [0.25, 0.4, 0.7, 1.0, 1.414]:
            for wd in [1e-2, 1e-3, 0.0]:
                runner.add(tasks, 1e-6, ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}',
                                                     lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                                     selection_method='random', base_kernel='linear',
                                                     kernel_transforms=[]))
    for lr in [8e-2, 1e-1, 2e-1]:
        for sigma_w in [0.25, 0.4, 0.5]:
            for wd in [1e-2, 1e-3, 1e-1, 0.0, 3e-3]:
                runner.add(tasks, 1e-6, ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}',
                                                     lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                                     selection_method='random', base_kernel='linear',
                                                     kernel_transforms=[]))

    for lr in [2e-1, 3e-1, 4e-1]:
        for sigma_w in [0.25]:
            for wd in [0.0, 1e-3, 3e-3]:
                runner.add(tasks, 1e-6, ModelTrainer(f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}',
                                                     lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                                     selection_method='random', base_kernel='linear',
                                                     kernel_transforms=[]))

    for lr in [7.5e-2]:
        for sigma_w in [1.0]:
            for wd in [0.0]:
                for wig in [0.25]:
                    for sigma_b in [0.1, 0.4, 1.0]:
                        runner.add(tasks, 1e-6,
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
                        runner.add(tasks, 1e-6,
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
                        runner.add(tasks, 1e-6,
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
                runner.add(tasks, 1e-6,
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
            runner.add(tasks, 1e-6,
                       ModelTrainer(
                           f'NN_lr-{lr:g}_sigmaw-{sigma_w:g}_wd-{wd:g}_wig-{wig:g}-sigmab-{sigma_b:g}-{lr_sched}',
                           lr=lr, weight_decay=wd, weight_gain=sigma_w,
                           weight_init_gain=wig, bias_gain=sigma_b, lr_sched=lr_sched,
                           selection_method='random', base_kernel='linear',
                           kernel_transforms=[]))


def add_silu_tuning_configs_to_runner(runner: JobRunner, tasks: List[Task]):
    for sigma_w in [0.2, 0.5, 1.0]:
        for sigma_b in [0.1, 0.25, 0.5, 1.0]:
            for lr in [3e-2 / sigma_w, 5e-2 / sigma_w, 7.5e-2 / sigma_w, 1e-1 / sigma_w]:
                wd = 0.0
                wig = 1.0
                runner.add(tasks, 1e-6,
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
        runner.add(tasks, 1e-6,
                   ModelTrainer(f'NN_sigmaw-{sigma_w:g}_wd-{wd:g}_sigmab-{sigma_b:g}_lr-{lr:g}',
                                lr=lr, weight_decay=wd, weight_gain=sigma_w,
                                weight_init_gain=wig, bias_gain=sigma_b, act='silu',
                                selection_method='random', base_kernel='linear',
                                kernel_transforms=[]))


def run_experiments(exp_name: str, n_splits: int, add_configs_func: Callable[[JobRunner, List[Task]], None],
                    batch_sizes_configs: Optional[List[List[int]]] = None, task_descs: Optional[List[str]] = None,
                    filter_alg_names: Optional[List[str]] = None, use_pool_for_normalization: bool = True,
                    max_jobs_per_device: int = 20, n_train_initial: int = 256, ds_names: Optional[List[str]] = None,
                    sequential_split: Optional[int] = 9):
    """
    This function allows to run experiments in a parallelized fashion.
    :param exp_name: Name for the group of experiments. This name will be used as a folder name.
    :param n_splits: Number of random splits to run.
    :param add_configs_func: Callback function that adds the desired experiment configurations to a JobRunner.
    :param batch_sizes_configs: Optional list of lists of batch sizes.
    The callback function will be called once for each list of batch sizes,
    with tasks using this list of batch sizes for BMDAL.
    By default, batch_sizes_configs=[[256]*16] will be used as in the paper.
    :param task_descs: Optional list of task descriptions, which will be appended to the dataset names.
    One task description per list of batch sizes in batch_sizes_configs should be provided.
    :param filter_alg_names: Optional list of strings that specifies a list of alg_names.
    If provided, configurations added by the callback function whose alg_name is not in this list will be ignored.
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

    for current_n_splits in range(1, n_splits + 1):
        # run each split sequentially
        print(f'Running all configurations on split {current_n_splits - 1}')
        # run only one experiment per GPU on split 9 for timing experiments
        scheduler = JobScheduler(max_jobs_per_device=1 if current_n_splits-1 == sequential_split
                                                       else max_jobs_per_device)
        runner = JobRunner(scheduler=scheduler, n_splits=current_n_splits, exp_name=exp_name,
                           filter_alg_names=filter_alg_names,
                           use_pool_for_normalization=use_pool_for_normalization)
        for batch_sizes_config, task_desc in zip(batch_sizes_configs, task_descs):
            tasks = Task.get_tabular_tasks(n_train=n_train_initial, al_batch_sizes=batch_sizes_config,
                                           ds_names=ds_names,
                                           desc=task_desc)
            add_configs_func(runner, tasks)
        runner.run_all()


def add_configs_relu(runner: JobRunner, tasks: List[Task]):
    add_all_configs_to_runner(runner, tasks, post_sigma=1e-2, maxdet_sigma=1e-2, weight_gain=0.2, bias_gain=0.2,
                              lr=0.375, act='relu')


def add_configs_silu(runner: JobRunner, tasks: List[Task]):
    add_all_configs_to_runner(runner, tasks, weight_gain=0.5, bias_gain=1.0, post_sigma=1e-2, maxdet_sigma=1e-2,
                              lr=0.15, act='silu')


if __name__ == '__main__':
    # TODO: we used use_pool_for_normalization = False in our experiments (since this was the original implementation).
    #  If you want to directly compare to our experiments, you should also use the same setting.
    #  However, if you want to run new experiments, it may be better to set it to True,
    #  especially if you want to start from a small initial training set
    use_pool_for_normalization = False

    # ReLU experiments
    run_experiments('relu', 20, add_configs_relu,
                    use_pool_for_normalization=use_pool_for_normalization)
    # ReLU batch size experiments
    run_experiments('relu', 20, add_configs_relu,
                    batch_sizes_configs=[[2**(12-m)]*(2**m) for m in range(7) if m != 4],
                    task_descs=[f'{2**(12-m)}x{2**m}' for m in range(7) if m != 4],
                    filter_alg_names=['NN_lcmd-tp_grad_rp-512', 'NN_kmeanspp-p_grad_rp-512_acs-rf-512',
                                      'NN_fw-p_grad_rp-512_acs-rf-hyper-512', 'NN_maxdist-p_grad_rp-512_train',
                                      'NN_maxdet-p_grad_rp-512_train', 'NN_maxdiag_grad_rp-512_acs-rf-512'],
                    use_pool_for_normalization=use_pool_for_normalization)
    # SiLU experiments, without batch size experiments
    run_experiments('silu', 20, add_configs_silu,
                    use_pool_for_normalization=use_pool_for_normalization)

    # for hyperparameter optimization
    # run_experiments('relu_tuning', 2, add_relu_tuning_configs_to_runner,
    #                 use_pool_for_normalization=use_pool_for_normalization)
    # run_experiments('silu_tuning', 2, add_silu_tuning_configs_to_runner,
    #                 use_pool_for_normalization=use_pool_for_normalization)
