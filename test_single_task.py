from .train import *
from .data import *
from .task_execution import get_devices
import time


def run_single_task():
    """
    Test function for running Batch Active Learning on a single task.
    """
    task = Task.get_tabular_tasks(n_train=256, al_batch_sizes=[256] * 16, ds_names=['kegg_undir_uci'])[0]
    task_split = TaskSplit(task, id=0)
    sigma = 1e-3
    compute_eff_dim = True
    kwargs = dict(maxdet_sigma=sigma, bait_sigma=sigma, compute_eff_dim=compute_eff_dim, allow_float64=True,
                  lr=0.375, weight_gain=0.2, bias_gain=0.2, lr_sched='lin'
                  )
    trainer = ModelTrainer('NN_test', act='relu',
                           base_kernel='grad',
                           kernel_transforms=[
                               # ('scale', [None]),
                               ('rp', [2048]),
                               # ('acs-rf-hyper', [512, None]),
                               # ('acs-rf', [512]),
                               ('train', [sigma, None]),
                           ], selection_method='bait',
                           sel_with_train=True,
                           allow_maxdet_fs=False,
                           # print_effective_dimension=True,
                           n_epochs=256, n_models=1, al_on_cpu=False, **kwargs)
    # trainer = ModelTrainer('NN_test', act='relu',
    #                        base_kernel='ll',
    #                        kernel_transforms=[
    #                            # ('scale', [None]),
    #                            ('rp', [512]),
    #                            # ('acs-rf-hyper', [512, None]),
    #                            # ('acs-rf', [512]),
    #                            ('train', [sigma, None]),
    #                        ], selection_method='bait', overselection_factor=2.0,
    #                        sel_with_train=False,
    #                        allow_maxdet_fs=False,
    #                        # print_effective_dimension=True,
    #                        n_epochs=256, n_models=1, al_on_cpu=False, **kwargs)
    start_time = time.time()
    results = trainer(task_split, device=get_devices()[0], do_timing=False)
    print(results)
    print(f'Time: {time.time() - start_time:g}')


if __name__ == '__main__':
    run_single_task()
    # import cProfile
    # cProfile.run('run_single_task()')
    pass
