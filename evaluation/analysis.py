import numpy as np
from typing import *
from pathlib import Path
import os

from .. import custom_paths
from .. import utils

class ExperimentResults:
    def __init__(self, results_dict: dict, exp_name: str):
        # usually, results_dict[alg_name][task_name][split_idx] = something
        self.results_dict = results_dict
        self.exp_name = exp_name
        self.alg_names = list(self.results_dict.keys())
        # self.alg_names.sort()
        self.task_names = list(set.union(*[set(alg_results.keys()) for alg_results in self.results_dict.values()]))
        self.task_names.sort()

    def map_single_split(self, f: Callable):
        # can be used e.g. to compute average log metric results
        return ExperimentResults({alg_name: {task_name: [f(split) for split in split_results]
                                             for task_name, split_results in task_results.items()}
                                  for alg_name, task_results in self.results_dict.items()}, exp_name=self.exp_name)

    def get_learning_curves(self, key: str) -> 'ExperimentResults':
        return ExperimentResults({alg_name: {task_name: np.mean(np.log([split['errors'][key] for split in split_results]), axis=0)
                                             for task_name, split_results in task_results.items() if int(task_name.split('_')[-1].split('x')[0]) == 256}
                                  for alg_name, task_results in self.results_dict.items()}, exp_name=self.exp_name)

    def get_avg_al_stats(self, key: str) -> 'ExperimentResults':
        return self.map_single_split(
            lambda split_dict: [stat[key] for stat in split_dict['al_stats'] if key in stat])

    def get_avg_al_times(self, key: str) -> 'ExperimentResults':
        return self.map_single_split(
            lambda split_dict: [stat[key]['total'] for stat in split_dict['al_stats'] if key in stat])

    def get_avg_errors(self, key: str, use_log: bool = True) -> 'ExperimentResults':
        if use_log:
            return self.map_single_split(
                lambda split_dict: np.mean(np.log(split_dict['errors'][key][1:])))
        else:
            return self.map_single_split(
                lambda split_dict: np.mean(split_dict['errors'][key][1:]))

    def get_last_errors(self, key: str, use_log: bool = True) -> 'ExperimentResults':
        if use_log:
            return self.map_single_split(
                lambda split_dict: np.mean(np.log(split_dict['errors'][key][-1])))
        else:
            return self.map_single_split(
                lambda split_dict: np.mean(split_dict['errors'][key][-1]))

    def get_average_al_times(self) -> 'ExperimentResults':
        return self.map_single_split(
            lambda split_dict: np.mean([sr['kernel_time']['total'] + sr['selection_time']['total']
                                        for sr in split_dict['al_stats']]))

    def select_split(self, i: int) -> 'ExperimentResults':
        for alg_name, task_results in self.results_dict.items():
            for task_name, split_results in task_results.items():
                if len(split_results) <= i:
                    print(f'Invalid index for alg {alg_name} on task {task_name}')
        return ExperimentResults({alg_name: {task_name: split_results[i]
                                             for task_name, split_results in task_results.items()}
                                  for alg_name, task_results in self.results_dict.items()}, exp_name=self.exp_name)

    def filter_task_suffix(self, task_suffix: str) -> 'ExperimentResults':
        return ExperimentResults(
            {alg_name: {task_name: task_dict
                        for task_name, task_dict in alg_dict.items() if task_name.endswith(task_suffix)}
            for alg_name, alg_dict in self.results_dict.items()}, exp_name=self.exp_name)

    def filter_task_prefix(self, task_prefix: str) -> 'ExperimentResults':
        return ExperimentResults(
            {alg_name: {task_name: task_dict
                        for task_name, task_dict in alg_dict.items() if task_name.startswith(task_prefix)}
            for alg_name, alg_dict in self.results_dict.items()}, exp_name=self.exp_name)

    def filter_task_names(self, task_names: List[str]) -> 'ExperimentResults':
        return ExperimentResults(
            {alg_name: {task_name: task_dict
                        for task_name, task_dict in alg_dict.items() if task_name in task_names}
            for alg_name, alg_dict in self.results_dict.items()}, exp_name=self.exp_name)

    def filter_alg_names(self, alg_names: Iterable[str]) -> 'ExperimentResults':
        return ExperimentResults({alg_name: self.results_dict[alg_name] for alg_name in alg_names
                                  if alg_name in self.results_dict}, exp_name=self.exp_name)

    def filter_common_algs(self) -> 'ExperimentResults':
        common_alg_names = [alg_name for alg_name, alg_dict in self.results_dict.items()
                            if set(alg_dict.keys()) == set(self.task_names)]
        return self.filter_alg_names(common_alg_names)

    def analyze_errors(self):
        n_steps = 0
        for alg_name, task_results in self.results_dict.items():
            for task_name, split_results in task_results.items():
                for split_idx, split_result in enumerate(split_results):
                    for al_stat_idx, al_stat in enumerate(split_result['al_stats']):
                        n_steps += 1
                        if 'selection_status' in al_stat and al_stat['selection_status'] is not None:
                            print(f'Alg {alg_name} failed on step {al_stat_idx} of split {split_idx} of task {task_name}:',
                                  al_stat['selection_status'])

        print(f'Total number of DBAL steps across all experiments: {n_steps}')

    def analyze_eff_dims(self):
        n_larger = 0
        n_total = 0
        eff_dim_sum_grad = 0.0
        eff_dim_sum_ll = 0.0
        for alg_name in self.results_dict:
            if not alg_name.endswith('_grad_rp-512'):
                continue
            alg_name_ll = alg_name.replace('_grad_rp-512', '_ll')
            for task_name, split_results in self.results_dict[alg_name].items():
                if not task_name.endswith('256x16'):
                    continue
                for split_idx, split_result in enumerate(split_results):
                    for al_stat_idx, al_stat in enumerate(split_result['al_stats']):
                        eff_dim = al_stat['eff_dim']
                        al_stat_ll = self.results_dict[alg_name_ll][task_name][split_idx]['al_stats'][al_stat_idx]
                        eff_dim_ll = al_stat_ll['eff_dim']
                        n_total += 1
                        eff_dim_sum_grad += eff_dim
                        eff_dim_sum_ll += eff_dim_ll
                        if eff_dim > eff_dim_ll:
                            n_larger += 1

        print(f'eff dim was larger for grad_rp-512 than for ll in {100*n_larger/n_total:g}% of cases')
        print(f'avg eff dim for grad_rp-512: {eff_dim_sum_grad/n_total:g}')
        print(f'avg eff dim for ll: {eff_dim_sum_ll/n_total:g}')

    @staticmethod
    def load(exp_name: str) -> 'ExperimentResults':
        results_path = Path(custom_paths.get_results_path()) / exp_name
        pkl_filename = Path(custom_paths.get_cache_path()) / exp_name / 'results.pkl'
        results = None
        # first try to load from cached pkl file
        if utils.existsFile(pkl_filename) \
            and os.path.getmtime(pkl_filename) >= utils.last_mod_time_recursive(str(results_path)):
            try:
                results = utils.deserialize(pkl_filename)
            except Exception as e:
                print(f'Received exception while trying to load cached results, '
                      f'reloading results without cache. Exception: {e}')

        # if loading cached data did not work, load from scratch
        if results is None:
            results = {}
            for task_path in results_path.iterdir():
                task_name = task_path.name
                for alg_path in task_path.iterdir():
                    alg_name = alg_path.name
                    alg_results = {}
                    split_exists = False
                    for split_path in sorted(alg_path.iterdir(), key=lambda path: int(path.name)):
                        results_file = split_path / 'results.json'
                        if utils.existsFile(results_file):
                            split_results = utils.deserialize(split_path / 'results.json', use_json=True)
                            if task_name in alg_results:
                                alg_results[task_name].append(split_results)
                            else:
                                alg_results[task_name] = [split_results]
                            split_exists = True

                    if split_exists:
                        if alg_name in results:
                            results[alg_name].update(alg_results)
                        else:
                            results[alg_name] = alg_results
            utils.serialize(pkl_filename, results)
        return ExperimentResults(results_dict=results, exp_name=exp_name)


def get_latex_metric_name(metric_name: str) -> str:
    conversion_dict = {'mae': r'MAE',
                       'rmse': r'RMSE',
                       'maxe': r'MAXE',
                       'q95': r'95\% quantile',
                       'q99': r'99\% quantile'
                       }
    return conversion_dict[metric_name]


def get_latex_ds_name(ds_name: str) -> str:
    conversion_dict = {'ct': 'ct_slices',
                       'kegg_undir_uci': 'kegg_undir',
                       'query_agg_count': 'query',
                       'road_network': 'road',
                       'wecs': 'wec_sydney'}
    if ds_name in conversion_dict:
        ds_name = conversion_dict[ds_name]
    return ds_name.replace('_', r'\_')


def get_latex_task(task: str) -> str:
    conversion_dict = {'online_video': r'online\_video',
                       'sgemm': r'sgemm',
                       'kegg_undir_uci': r'kegg\_undir',
                       'stock': r'stock',
                       'wecs': r'wec\_sydney',
                       'sarcos': r'sarcos',
                       'diamonds': r'diamonds',
                       'fried': r'fried',
                       'road_network': r'road',
                       'poker': r'poker',
                       'mlr_knn_rng': r'mlr\_knn\_rng',
                       'methane': r'methane',
                       'protein': r'protein',
                       'ct': r'ct\_slices',
                       'query_agg_count': r'query'
                       }
    return conversion_dict[task]


def get_latex_selection_method(selection_method: str) -> str:
    conversion_dict = {'random': r'\textsc{Random}',
                       'fw': r'\textsc{FrankWolfe}',
                       'bait-f': r'\textsc{Bait-F}',
                       'bait-fb': r'\textsc{Bait-FB}',
                       'kmeanspp': r'\textsc{KMeansPP}',
                       'lcmd': r'\textsc{LCMD}',
                       'maxdet': r'\textsc{MaxDet}',
                       'maxdist': r'\textsc{MaxDist}',
                       'maxdiag': r'\textsc{MaxDiag}'
                       }

    parts = selection_method.split('-')
    method_name = '-'.join(parts[:-1]) if len(parts) > 1 else selection_method
    result = conversion_dict[method_name]
    if len(parts) > 1:
        result += '-TP' if parts[-1] == 'tp' else '-P'
    if method_name == 'lcmd':
        result += ' (ours)'
    return result


def get_latex_kernel(base_kernel: str, kernel_transformations: List[Tuple[str, List]], n_models: int) -> str:
    conversion_base_kernel_dict = {'grad': r'\mathrm{grad}',
                                   'll': r'\mathrm{ll}',
                                   'linear': r'\mathrm{lin}',
                                   'nngp': r'\mathrm{nngp}'}

    steps = [conversion_base_kernel_dict[base_kernel]]

    for name, args in kernel_transformations:
        if name == 'train':
            steps.append(r'\mathcal{X}_{\operatorname{train}}')
        elif name == 'scale':
            steps.append(r'\operatorname{scale}(\mathcal{X}_{\operatorname{train}})')
        elif name == 'rp':
            steps.append(r'\operatorname{rp}(' + str(args[0]) + ')')
        elif name == 'ens':
            steps.append(r'\operatorname{ens}(' + str(n_models) + ')')
        elif name == 'acs-rf':
            steps.append(r'\operatorname{acs-rf}(' + str(args[0]) + ')')
        elif name == 'acs-rf-hyper':
            steps.append(r'\operatorname{acs-rf-hyper}(' + str(args[0]) + ')')
        elif name == 'acs-grad':
            steps.append(r'\operatorname{acs-grad}')
        else:
            raise ValueError(f'Unknown kernel transformation "{name}"')

    return '$k_{' + r' \to '.join(steps) + '}$'


def save_latex_table_all_algs(results: ExperimentResults, filename: str):
    # creates a table for all algorithms, all metric are averaged over all data sets
    # Selection method | Kernel | MAE | RMSE | 95% | 99% | MAXE | avg. time

    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    all_alg_names = results.alg_names

    mae = results.get_avg_errors('mae')
    rmse = results.get_avg_errors('rmse')
    q95 = results.get_avg_errors('q95')
    q99 = results.get_avg_errors('q99')
    maxe = results.get_avg_errors('maxe')
    kernel_time = results.get_avg_al_times('kernel_time').select_split(9)
    selection_time = results.get_avg_al_times('selection_time').select_split(9)

    metrics_and_names = [('MAE', mae), ('RMSE', rmse), (r'95\%', q95), (r'99\%', q99), ('MAXE', maxe),
                         ('kernel_time', kernel_time), ('selection_time', selection_time)]
    alg_metrics = {alg_name: {name: np.mean([np.mean(metric.results_dict[alg_name][ds_name + '_256x16'])
                                             for ds_name in ds_names])
                    for name, metric in metrics_and_names} for alg_name in all_alg_names}

    n_digits = 3
    best_alg_names_per_metric = {}
    for metric_name, _ in metrics_and_names:
        rounded_results = [round(alg_metrics[alg_name][metric_name], n_digits) for alg_name in all_alg_names]
        min_result = np.min(rounded_results)
        best_alg_names_per_metric[metric_name] = [all_alg_names[i] for i in range(len(all_alg_names))
                                                  if rounded_results[i] == min_result]

    table_rows = {}
    table_header = '\\begin{tabular}{cccccccc}\n' + \
                   ' & '.join([r'Selection method', r'Kernel', r'MAE', r'RMSE', r'95\%', r'99\%',
                               r'MAXE', r'avg.\ time [$s$]']) + '\\\\\n\\hline\n'
    table_footer = '\n\\end{tabular}'

    # raw_sel_order = {'random': 0, 'maxdiag': 1, 'maxdet': 2, 'fw': 3, 'maxdist': 4, 'kmeanspp': 5, 'lcmd': 6}
    sel_name_order = ['random', 'maxdiag', 'maxdet', 'bait', 'fw', 'maxdist', 'kmeanspp', 'lcmd']

    raw_sel_names = {}

    for name_alg in all_alg_names:
        config = next(iter(results.results_dict[name_alg].values()))[0]['config']
        base_kernel = config['base_kernel']
        kernel_transformations = config['kernel_transforms']
        n_models = config.get('n_models', 1)

        raw_sel_name = name_alg.split('_')[1].split('-')[0]
        sel_name = get_latex_selection_method(name_alg.split('_')[1])
        raw_sel_names[name_alg] = raw_sel_name
        kernel_name = get_latex_kernel(base_kernel, kernel_transformations, n_models=n_models)
        if raw_sel_name == 'random':
            kernel_name = '---'
        alg_results = []
        for metric_name, _ in metrics_and_names[:5]:
            value_str = f'{alg_metrics[name_alg][metric_name]:5.3f}'
            if name_alg in best_alg_names_per_metric[metric_name]:
                value_str = r'\textbf{' + value_str + r'}'
            alg_results.append(value_str)
        alg_time = alg_metrics[name_alg]['kernel_time'] + alg_metrics[name_alg]['selection_time']

        row_strs = [sel_name, kernel_name] + alg_results + [f'{alg_time:5.3f}']
        table_rows[name_alg] = ' & '.join(row_strs)

    sub_groups = [[alg_name for alg_name in all_alg_names if raw_sel_names[alg_name] == sel_name] for sel_name in sel_name_order]
    sub_groups = [sub_group for sub_group in sub_groups if len(sub_group) > 0]
    for i in range(len(sub_groups)):
        sub_groups[i].sort(key=lambda alg_name: alg_metrics[alg_name]['RMSE'])

    sub_group_strs = ['\\\\\n'.join([table_rows[alg_name] for alg_name in sub_group]) for sub_group in sub_groups]

    result_str = table_header + ' \\\\\n\\hline\n'.join(sub_group_strs) + table_footer

    utils.writeToFile(Path(custom_paths.get_plots_path()) / results.exp_name / filename, result_str)


def save_latex_table_data_sets(results: ExperimentResults, filename: str, use_log: bool = True,
                               use_last_error: bool = False, metric_name: str = 'rmse'):
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    ds_names.sort()
    alg_names = results.alg_names

    if use_last_error:
        rmse = results.get_last_errors(metric_name, use_log=use_log)
    else:
        rmse = results.get_avg_errors(metric_name, use_log=use_log)
    alg_metrics = {alg_name: {ds_name: np.mean(rmse.results_dict[alg_name][ds_name + '_256x16'])
                              for ds_name in ds_names} for alg_name in alg_names}

    n_digits = 3
    best_alg_names_per_data_set = {}
    for ds_name in ds_names:
        rounded_results = [round(alg_metrics[alg_name][ds_name], n_digits) for alg_name in alg_names]
        min_result = np.min(rounded_results)
        best_alg_names_per_data_set[ds_name] = [alg_names[i] for i in range(len(alg_names)) if
                                                rounded_results[i] == min_result]

    table_list = []
    table_header = '\\begin{tabular}{' + ('c' * (len(alg_names)+1)) + '}\n' + \
                   ' & '.join([r'Data set'] + [get_latex_selection_method(alg_name.split('_')[1])
                                               for alg_name in alg_names]) + '\\\\\n\\hline\n'
    table_footer = '\n\\end{tabular}'

    for ds_name in ds_names:
        ds_results = []
        for name_alg in alg_names:
            value_str = f'{alg_metrics[name_alg][ds_name]:5.3f}'
            if name_alg in best_alg_names_per_data_set[ds_name]:
                value_str = r'\textbf{' + value_str + r'}'
            ds_results.append(value_str)

        row_strs = [get_latex_task(ds_name)] + ds_results
        table_list.append(' & '.join(row_strs))

    table_list = table_list

    result_str = table_header + ' \\\\\n'.join(table_list) + table_footer

    utils.writeToFile(Path(custom_paths.get_plots_path()) / results.exp_name / filename, result_str)


def print_single_task_results(exp_results: ExperimentResults, task_name: str):
    """
    Prints results for a single task. For each log metric value, the function prints
    mean +- standard error (estimated standard deviation of the mean estimator,
    when considering the data sets to be fixed)
    :param exp_results: ExperimentResults object whose results should be printed.
    :param task_name: Task for which the results should be printed.
    """
    metric_names = ['mae', 'rmse', 'q95', 'q99', 'maxe']

    print(f'Results for task {task_name}:')
    exp_results = exp_results.filter_task_names([task_name]).filter_common_algs()
    alg_names = exp_results.alg_names
    for metric_name in metric_names:
        avg_results = exp_results.get_avg_errors(metric_name)
        results = avg_results.results_dict
        print(f'Results for metric {metric_name}:')
        alg_means = {alg_name: np.mean(results[alg_name][task_name]) for alg_name in alg_names}
        alg_stds = {alg_name: np.std(results[alg_name][task_name])/np.sqrt(len(results[alg_name][task_name]))
                    for alg_name in alg_names}

        alg_names_sorted = utils.dict_argsort(alg_means)
        str_table = []
        for alg_name in alg_names_sorted:
            str_table.append([alg_name + ': ', f'{alg_means[alg_name]:5.3f} ', '+- ', f'{alg_stds[alg_name]:5.3f}'])
        print(utils.pretty_table_str(str_table))
        print()

    print('\n\n')


def print_all_task_results(exp_results: ExperimentResults):
    for task_name in exp_results.task_names:
        print_single_task_results(exp_results, task_name)


def print_avg_results(exp_results: ExperimentResults, relative_to: Optional[str] = None,
                      filter_suffix: str = '256x16'):
    """
    Prints experiment results averaged over splits, steps and data sets.
    :param exp_results: ExperimentResults object whose results should be printed.
    :param relative_to: alg_name (or part thereof) of a method that results should be compared to.
    If None, the log metric values are printed directly +- the standard error where the data sets are considered fixed.
    Otherwise, the difference in log metric values between the respective method and the relative_to method is printed,
    +- the standard error thereof where the data sets are considered to be randomly drawn.
    In the latter case, the standard error is going to be larger
    but indicates how strongly the relative performance varies across data sets.
    :param filter_suffix: Only tasks with this suffix are going to be analyzed.
    """
    if relative_to is not None:
        print(f'Averaged results across tasks, relative to {relative_to}:')
    else:
        print('Averaged results across tasks:')

    exp_results = exp_results.filter_task_suffix(filter_suffix).filter_common_algs()

    metric_names = ['mae', 'rmse', 'q95', 'q99', 'maxe']

    # find all algorithms that occur in all tasks
    alg_names = exp_results.alg_names
    task_names = exp_results.task_names
    relative_to_name = None if relative_to is None else \
        (relative_to if relative_to in alg_names
        else [name for name in alg_names if relative_to in name][0])

    for metric_name in metric_names:
        print(f'Results for metric {metric_name}:')
        avg_results = exp_results.get_avg_errors(metric_name)
        alg_task_means = {alg_name: {task_name: np.mean(avg_results.results_dict[alg_name][task_name])
                                     for task_name in task_names}
                          for alg_name in alg_names}
        if relative_to is not None:
            alg_task_means = {alg_name: {task_name: alg_task_means[alg_name][task_name]
                                                    - alg_task_means[relative_to_name][task_name]
                                         for task_name in task_names}
                              for alg_name in alg_names}
        alg_means = {alg_name: np.mean(list(alg_task_means[alg_name].values())) for alg_name in alg_names}
        if relative_to is not None:
            # take std over tasks
            alg_stds = {alg_name: np.std(list(task_means.values())) / np.sqrt(len(list(task_means.values())))
                        for alg_name, task_means in alg_task_means.items()}
        else:
            # take sd over splits
            alg_stds = {alg_name: np.linalg.norm([np.std(avg_results.results_dict[alg_name][task_name])
                                                  / np.sqrt(len(avg_results.results_dict[alg_name][task_name]))
                                                  for task_name in task_names]) / len(task_names)
                        for alg_name in alg_names}

        alg_names_sorted = utils.dict_argsort(alg_means)
        str_table = []
        for alg_name in alg_names_sorted:
            str_table.append([alg_name + ': ', f'{alg_means[alg_name]:5.3f} ', '+- ', f'{alg_stds[alg_name]:5.3f}'])
        print(utils.pretty_table_str(str_table))
        print()

    print('\n\n')
