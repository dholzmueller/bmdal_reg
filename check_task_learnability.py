from multiprocessing.pool import Pool
import multiprocessing as mp

from .train import *
from .data import *
from .task_execution import get_devices

# This file can be used to check how much the error for random selection drops on different data sets
# when increasing the training set size.
# We used this to remove data sets where increasing the number of samples does not have a strong effect.


def single_learnability(args):
    task, task_id, device_idx = args
    # task = Task.get_tabular_tasks(n_train=10, al_batch_sizes=[2**k for k in range(9)] + [256]*10)[3]
    trainer = ModelTrainer('NN_test', save=False)
    return trainer(TaskSplit(task, id=task_id), device=get_devices()[device_idx])


def check_task_learnability(n_repeats=5, n_train_small=1000, n_train_large=5000, task_names=None,
                            n_processes_per_device=5):
    small_tasks = Task.get_tabular_tasks(n_train=n_train_small, al_batch_sizes=[], ds_names=task_names)
    large_tasks = Task.get_tabular_tasks(n_train=n_train_large, al_batch_sizes=[], ds_names=task_names)
    n_devices = len(get_devices())
    # task = Task.get_tabular_tasks(n_train=10, al_batch_sizes=[2**k for k in range(9)] + [256]*10)[3]
    # trainer = ModelTrainer('NN_test', save=False, batch_size=512)
    for small_task, large_task in zip(small_tasks, large_tasks):
        assert(small_task.task_name == large_task.task_name)
        pool = Pool(processes=min(n_repeats, n_devices * n_processes_per_device))
        small_results = pool.map(single_learnability,
                                 [(small_task, task_id, task_id % n_devices) for task_id in range(n_repeats)],
                                 chunksize=1)
        pool = Pool(processes=min(n_repeats, n_devices * n_processes_per_device))
        large_results = pool.map(single_learnability,
                                 [(large_task, task_id, task_id % n_devices) for task_id in range(n_repeats)],
                                 chunksize=1)
        # small_results = [trainer(TaskSplit(small_task, id=task_id), device=get_devices()[device_idx]) for task_id in range(n_repeats)]
        # large_results = [trainer(TaskSplit(large_task, id=task_id), device=get_devices()[device_idx]) for task_id in range(n_repeats)]
        small_errors = [r['errors']['rmse'][0] for r in small_results]
        large_errors = [r['errors']['rmse'][0] for r in large_results]
        small_mean_rmse = np.mean(small_errors)
        large_mean_rmse = np.mean(large_errors)
        gain_percent = 100 * (small_mean_rmse - large_mean_rmse) / small_mean_rmse
        print(f'Learnability on task {small_task.task_name}: Reduction by {gain_percent:g}% '
              f'from RMSE={np.mean(small_errors):g} +- {np.std(small_errors):g} for n_train={n_train_small} '
              f'to RMSE={np.mean(large_errors):g} +- {np.std(large_errors):g} for n_train={n_train_large}')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # check_task_learnability()
    # check_task_learnability(n_repeats=20)
    # check_task_learnability(n_repeats=20, task_names=['poker'])
    # check_task_learnability(n_repeats=20, n_train_small=512, n_train_large=512+16*256)
    # check_task_learnability(n_repeats=20, n_train_small=1024, n_train_large=1024+16*256)
    check_task_learnability(n_repeats=20, n_train_small=256, n_train_large=256+16*256)
    pass
