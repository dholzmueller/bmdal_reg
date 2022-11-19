import matplotlib
#matplotlib.use('Agg')
matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 10.95,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'legend.framealpha': 0.5,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}'
})
# from tueplots import bundles, fonts, fontsizes, figsizes
# matplotlib.rcParams.update(bundles.jmlr2001())
# matplotlib.rcParams.update(fonts.jmlr2001_tex())
# matplotlib.rcParams.update(fontsizes.jmlr2001())

import matplotlib.pyplot as plt
from .analysis import *
from pathlib import Path
import seaborn as sns

fontsize = 10
axis_font = {'size': str(fontsize)}
sns.axes_style("whitegrid")


def escape(name: str):
    return name.replace('_', r'\_')


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
          u'#bcbd22', u'#1f77b4']
markers = ['^', 'v', '<', '>', 'P', 'D', 's', 'v']


def plot_batch_sizes_ax(ax: plt.Axes, results: ExperimentResults, metric_name: str, set_ticks_and_labels: bool = True,
                        **plot_options):
    last_errors = results.get_last_errors(metric_name)
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})

    # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
    # default matplotlib colors:
    # colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
    #           u'#bcbd22', u'#17becf']

    color_idx = 0

    plot_options = utils.update_dict(dict(alpha=1.0, markersize=3.5), plot_options)

    for alg_name in results.alg_names:
        log_means = []
        alg_stds = []

        ds_batch_sizes = []

        for ds_name in ds_names:
            keys = []
            batch_sizes = []

            for key in last_errors.results_dict[alg_name]:
                if key.startswith(ds_name):
                    keys.append(key)
                    batch_sizes.append(int(key.split('_')[-1].split('x')[0]))

            batch_sizes = np.array(list(batch_sizes))
            idxs = np.argsort(batch_sizes)
            batch_sizes = batch_sizes[idxs]
            results_list = [np.mean(last_errors.results_dict[alg_name][key]) for key in keys]
            task_stds = [np.std(last_errors.results_dict[alg_name][key], axis=0) /
                        np.sqrt(len(last_errors.results_dict[alg_name][key])-1) for key in keys]
            alg_stds.append(np.asarray(task_stds)[idxs])
            log_means.append(np.asarray(results_list)[idxs])
            ds_batch_sizes.append(batch_sizes)

        if len(ds_batch_sizes) == 0:
            raise ValueError(f'No data set results available for alg {alg_name}')

        if alg_name == 'NN_random':
            # all batch sizes should provide equivalent results
            minus = np.mean(log_means) - np.linalg.norm(alg_stds) / len(alg_stds)
            plus = np.mean(log_means) + np.linalg.norm(alg_stds) / len(alg_stds)
            ax.axhspan(minus, plus, alpha=0.2, facecolor='k', edgecolor=None)
            ax.axhline(y=np.mean(log_means), ls='--', color='k',
                        label=get_latex_selection_method(alg_name.split('_')[1]))
        else:
            minus = np.mean(log_means, axis=0) - np.linalg.norm(alg_stds, axis=0) / len(alg_stds)
            plus = np.mean(log_means, axis=0) + np.linalg.norm(alg_stds, axis=0) / len(alg_stds)
            ax.fill_between(np.log(ds_batch_sizes[0]), minus, plus, facecolor=colors[color_idx], alpha=0.2)
            ax.plot(np.log(ds_batch_sizes[0]), np.mean(log_means, axis=0), '--', marker=markers[color_idx],
                    label=get_latex_selection_method(alg_name.split('_')[1]), color=colors[color_idx], **plot_options)
            color_idx += 1

    if set_ticks_and_labels:
        xlocs = [np.log(64), np.log(128), np.log(256), np.log(512), np.log(1024), np.log(2048), np.log(4096)]
        xlabels = ('64', '128', '256', '512', '1024', '2048', '4096')

        ax.set_xticks(xlocs)
        ax.set_xticklabels(xlabels)

        ax.set_xlabel(r'Acquisition batch size $N_{\mathrm{batch}}$', **axis_font)
        ax.set_ylabel(r'mean log ' + f'{get_latex_metric_name(metric_name)}', **axis_font)


def plot_batch_sizes(results: ExperimentResults, filename: Union[str, Path], metric_name: str,
                     figsize: Optional[Tuple[float, float]] = None):
    # plot mean final log metric against the al batch size
    fig, ax = plt.subplots(figsize=figsize or (4, 4))

    plot_batch_sizes_ax(ax, results, metric_name)

    ax.legend()
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_multiple_batch_sizes(results: ExperimentResults, filename: Union[str, Path],
                                  metric_names: List[str]):
    # plot averaged learning curve for all tasks
    fig, axs = plt.subplots(1, len(metric_names), figsize=(7, 4))

    for i, metric_name in enumerate(metric_names):
        plot_batch_sizes_ax(axs[i], results, metric_name)

    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=4)
    plt.tight_layout(rect=[0, 0.15, 1.0, 1.0])
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_batch_sizes_individual(results: ExperimentResults, metric_name: str):
    # plot mean final log metric against the al batch size
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    ds_names.sort()

    for ds_name in ds_names:
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_batch_sizes_ax(ax, results.filter_task_prefix(ds_name), metric_name)

        ax.legend()
        plt.tight_layout()
        plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / 'ds_batch_sizes' / \
                    f'{ds_name}_{metric_name}.pdf'
        utils.ensureDir(plot_name)
        plt.savefig(plot_name)
        plt.close(fig)


def plot_batch_sizes_individual_subplots(results: ExperimentResults, filename: Union[str, Path], metric_name: str):
    # plot mean final log metric against the al batch size
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    ds_names.sort()

    fig, axs = plt.subplots(5, 3, figsize=(9, 12))
    if len(ds_names) != 15:
        print(f'plot_batch_sizes_individual_subplots needs 15 data sets, but got {len(ds_names)} data sets')
        return
    for ds_idx, ds_name in enumerate(ds_names):
        i = ds_idx // 3
        j = ds_idx % 3
        ax = axs[i, j]
        plot_batch_sizes_ax(ax, results.filter_task_prefix(ds_name), metric_name, set_ticks_and_labels=False,
                            markersize=3.5)

        xlocs = [np.log(64), np.log(128), np.log(256), np.log(512), np.log(1024), np.log(2048), np.log(4096)]
        xlabels = ('64', '128', '256', '512', '1024', '2048', '4096')

        ax.set_xticks(xlocs)
        ax.set_xticklabels(xlabels if i == 4 else [''] * len(xlocs))

        if i == 4:
            ax.set_xlabel(r'Acquisition batch size $N_{\mathrm{batch}}$', **axis_font)
        if j == 0:
            ax.set_ylabel(r'mean log ' + f'{get_latex_metric_name(metric_name)}', **axis_font)
        ax.set_title(get_latex_ds_name(ds_name))

    fig.legend(*axs[4, 1].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1.0, 1.0])
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_batch_sizes_metrics_subplots(results: ExperimentResults, filename: Union[str, Path]):
    # plot mean final log metric against the al batch size

    fig, axs = plt.subplots(3, 2, figsize=(8, 9))
    metric_names = ['mae', 'rmse', 'q95', 'q99', 'maxe']
    for metric_idx, metric_name in enumerate(metric_names):
        i = metric_idx // 2
        j = metric_idx % 2
        ax = axs[i, j]
        plot_batch_sizes_ax(ax, results, metric_name, set_ticks_and_labels=True)

    axs[-1, -1].axis('off')
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.78, 0.18), ncol=1)
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_learning_curves_ax(ax: plt.Axes, results: ExperimentResults, metric_name: str, with_random_final: bool = True,
                            set_ticks_and_labels: bool = True, labels: Optional[List[str]] = None, **plot_options):
    learning_curves = results.get_learning_curves(metric_name)

    # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
    # default matplotlib colors:
    # colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
    #           u'#bcbd22', u'#17becf']
    # colors = [u'#a06010', u'#d62728', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#e377c2', u'#7f7f7f',
    #           u'#bcbd22', u'#1f77b4']

    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})

    color_idx = 0

    for i, alg_name in enumerate(results.alg_names):
        log_means = [learning_curves.results_dict[alg_name][ds_name + '_256x16'] for ds_name in ds_names]
        results_list = np.mean(log_means, axis=0)
        n_train = np.asarray([256*(i+1) for i in range(len(results_list))])

        plot_options = utils.update_dict(dict(alpha=1.0, markersize=3.5), plot_options)

        label = get_latex_selection_method(alg_name.split('_')[1]) if labels is None else labels[i]

        if alg_name == 'NN_random':
            ax.plot(np.log(n_train), results_list, '--o', color='k', label=label, **plot_options)
            if with_random_final:
                ax.plot([np.log(n_train[0]), np.log(n_train[-1])], [results_list[-1], results_list[-1]],
                        '--', color='k', **plot_options)
        else:
            ax.plot(np.log(n_train), results_list, '--', marker=markers[color_idx], color=colors[color_idx],
                    label=label, **plot_options)
            color_idx += 1

    if set_ticks_and_labels:
        xlocs = (np.log(256), np.log(512), np.log(1024), np.log(2048), np.log(4096))
        xlabels = ('256', '512', '1024', '2048', '4096')

        ax.set_xticks(xlocs)
        ax.set_xticklabels(xlabels)

        ax.set_xlabel(r'Training set size $N_{\mathrm{train}}$', **axis_font)
        ax.set_ylabel(r'mean log ' + f'{get_latex_metric_name(metric_name)}', **axis_font)


def plot_learning_curves(results: ExperimentResults, filename: Union[str, Path], metric_name: str,
                         labels: Optional[List[str]] = None, figsize: Optional[Tuple[float, float]] = None):
    # plot averaged learning curve for all tasks
    fig, axs = plt.subplots(figsize=figsize or (3.5, 3.5))

    plot_learning_curves_ax(axs, results=results, metric_name=metric_name, labels=labels)

    axs.legend()
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_multiple_learning_curves(results: ExperimentResults, filename: Union[str, Path],
                                  metric_names: List[str]):
    # plot averaged learning curve for all tasks
    fig, axs = plt.subplots(1, len(metric_names), figsize=(7, 4))

    for i, metric_name in enumerate(metric_names):
        plot_learning_curves_ax(axs[i], results, metric_name=metric_name)

    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=4)
    plt.tight_layout(rect=[0, 0.15, 1.0, 1.0])
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_learning_curves_individual(results: ExperimentResults, metric_name: str):
    # plot individual learning curve for all tasks
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    ds_names.sort()

    for ds_name in ds_names:
        fig, axs = plt.subplots(figsize=(3.5, 3.5))
        plot_learning_curves_ax(axs, results.filter_task_names([ds_name + '_256x16']), metric_name=metric_name)
        axs.legend(fontsize=fontsize)
        plt.tight_layout()
        plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / 'ds_learning_curves' / \
                    f'{ds_name}_{metric_name}.pdf'
        utils.ensureDir(plot_name)
        plt.savefig(plot_name)
        plt.close(fig)


def plot_learning_curves_individual_subplots(results: ExperimentResults, filename: str, metric_name: str):
    # plot individual learning curve for all tasks
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    ds_names.sort()

    fig, axs = plt.subplots(5, 3, figsize=(9, 12))

    if len(ds_names) != 15:
        print(f'plot_batch_sizes_individual_subplots needs 15 data sets, but got {len(ds_names)} data sets')
        return
    for ds_idx, ds_name in enumerate(ds_names):
        i = ds_idx // 3
        j = ds_idx % 3
        ax = axs[i, j]
        plot_learning_curves_ax(ax, results.filter_task_names([ds_name + '_256x16']), metric_name=metric_name,
                                set_ticks_and_labels=False, alpha=0.8, markersize=3, linewidth=1.0, markeredgewidth=0.0)

        xlocs = [np.log(256), np.log(512), np.log(1024), np.log(2048), np.log(4096)]
        xlabels = ('256', '512', '1024', '2048', '4096')

        ax.set_xticks(xlocs)
        ax.set_xticklabels(xlabels if i == 4 else [''] * len(xlocs))

        if i == 4:
            ax.set_xlabel(r'Training set size $N_{\mathrm{train}}$', **axis_font)
        if j == 0:
            ax.set_ylabel(f'mean log {get_latex_metric_name(metric_name)}', **axis_font)
        ax.set_title(get_latex_ds_name(ds_name))

    fig.legend(*axs[4, 1].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    plt.tight_layout(rect=[0, 0.05, 1.0, 1.0])

    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_learning_curves_metrics_subplots(results: ExperimentResults, filename: Union[str, Path]):
    # plot mean final log metric against the al batch size
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 9))
    metric_names = ['mae', 'rmse', 'q95', 'q99', 'maxe']
    for metric_idx, metric_name in enumerate(metric_names):
        i = metric_idx // 2
        j = metric_idx % 2
        ax = axs[i, j]
        plot_learning_curves_ax(ax, results, metric_name)

    axs[-1, -1].axis('off')
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.78, 0.18), ncol=1)
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_correlation_between_methods(results: ExperimentResults, filename: Union[str, Path], metric_name: str):
    sns.color_palette("Paired")

    # plot mean final log metric against the al batch size
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in results.task_names})
    ds_names.sort()
    task_names = [ds_name + '_256x16' for ds_name in ds_names]
    alg_names = [alg_name for alg_name in results.alg_names if alg_name != 'NN_random']

    fig, axs = plt.subplots(len(alg_names)-1, len(alg_names)-1, figsize=(7.8, 7.8), sharex='col', sharey='row')

    last_errors = results.get_avg_errors(metric_name)

    all_results = {alg_name: {task_name: np.mean(last_errors.results_dict[alg_name][task_name])
                              for task_name in task_names} for alg_name in alg_names + ['NN_random']}

    max_result = 0.0
    min_result = 0.0
    alpha = 0.7
    markersize = 6

    # generated using https://github.com/taketwo/glasbey
    # c_list = [(0,0,0), (215,0,0), (140,60,255), (2,136,0), (0,172,199), (152,255,0), (255,127,209), (108,0,79), (255,165,48),
    #      (0,0,157), (134,112,104), (0,73,66), (79,42,0), (0,253,207), (188,183,255)]
    c_list = [(174,20,20), (0,85,239), (0,143,0), (239,91,255), (225,149,10), (0,184,196), (120,78,120), (255,110,128),
              (114,90,36), (148,155,255), (19,109,100), (145,180,125), (148,55,252), (202,16,130), (97,122,157)]

    c = [rgb_to_hex(c) for c in c_list]

    for i in range(len(alg_names)):
        for j in range(i+1, len(alg_names)):
            alg_name_i = alg_names[i]
            alg_name_j = alg_names[j]
            k = 0
            for task_name in task_names:
                result_x = all_results[alg_name_i][task_name] - all_results['NN_random'][task_name]
                result_y = all_results[alg_name_j][task_name] - all_results['NN_random'][task_name]
                max_result = np.max([result_x, result_y, max_result])
                min_result = np.min([result_x, result_y, min_result])
                if i == 0 and j == 1:
                    axs[j - 1, i].plot(result_x, result_y, 'o', alpha=alpha,
                                       label=get_latex_task(task_name.split('_256')[0]), markersize=markersize,
                                       color=c[k], markeredgewidth=0.0)
                    k += 1
                    handles, labels = axs[0, 0].get_legend_handles_labels()
                    axs[0, -1].legend(handles, labels, loc='upper right', borderaxespad=0, fontsize=fontsize)
                else:
                    axs[j - 1, i].plot(result_x, result_y, 'o', markersize=markersize, alpha=alpha, color=c[k],
                                       markeredgewidth=0.0)
                    k += 1

                if i == 0:
                    axs[j - 1, i].set_ylabel(get_latex_selection_method(alg_name_j.split('_')[1]), **axis_font)
                if j == len(alg_names)-1:
                    axs[j - 1, i].set_xlabel(get_latex_selection_method(alg_name_i.split('_')[1]), **axis_font)

    for i in range(len(alg_names)):
        for j in range(i+1, len(alg_names)):
            axs[j-1, i].plot([min_result, max_result], [min_result, max_result], 'k-')
            axs[j-1, i].plot([min_result, max_result], [0.0, 0.0], 'k--')
            axs[j - 1, i].plot([0.0, 0.0], [min_result, max_result], 'k--')

    for i in range(len(alg_names)-1):
        for j in range(i+1, len(alg_names)-1):
            axs[i, j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)


def plot_skewness_ax(ax: plt.Axes, results: ExperimentResults, metric_name: str, alg_name: str,
                     use_relative_improvement: bool = False):
    sns.color_palette("Paired")

    # plot mean final log metric against the al batch size
    all_alg_names = list(results.results_dict.keys())
    all_task_names = {task_name for alg_name in all_alg_names for task_name in results.results_dict[alg_name].keys()}
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in all_task_names})
    task_names = [ds_name + '_256x16' for ds_name in ds_names]
    task_names.sort()

    avg_errors = results.get_avg_errors(metric_name)
    random_name = 'NN_random'

    all_results = {name: {task_name: np.mean(avg_errors.results_dict[name][task_name])
                              for task_name in task_names} for name in [alg_name, random_name]}

    alpha = 0.7
    markersize = 6

    # generated using https://github.com/taketwo/glasbey
    # c_list = [(0,0,0), (215,0,0), (140,60,255), (2,136,0), (0,172,199), (152,255,0), (255,127,209), (108,0,79), (255,165,48),
    #      (0,0,157), (134,112,104), (0,73,66), (79,42,0), (0,253,207), (188,183,255)]
    c_list = [(174,20,20), (0,85,239), (0,143,0), (239,91,255), (225,149,10), (0,184,196), (120,78,120), (255,110,128),
              (114,90,36), (148,155,255), (19,109,100), (145,180,125), (148,55,252), (202,16,130), (97,122,157)]

    c = [rgb_to_hex(c) for c in c_list]

    x_values = []
    y_values = []

    for task_idx, task_name in enumerate(task_names):
        random_results = results.results_dict[random_name][task_name]
        n_splits = len(random_results)
        first_step_results = {metric: np.mean([np.log(random_results[i]['errors'][metric][0]) for i in range(n_splits)])
                              for metric in ['rmse', 'mae', 'q95', 'q99', 'maxe']}
        x_value = first_step_results['rmse'] - first_step_results['mae']
        y_value = all_results[alg_name][task_name] - all_results[random_name][task_name]
        if use_relative_improvement:
            # divide by (avg log rmse - initial log rmse)
            diff = all_results[random_name][task_name] - first_step_results[metric_name]
            y_value /= diff
        x_values.append(x_value)
        y_values.append(y_value)
        ax.plot(x_value, y_value, 'o', alpha=alpha, label=get_latex_task(task_name.split('_256')[0]),
                markersize=markersize, color=c[task_idx], markeredgewidth=0.0)
        sel_method_name = get_latex_selection_method(alg_name.split('_')[1])
        latex_metric_name = get_latex_metric_name(metric_name)
        if use_relative_improvement:
            ax.set_ylabel(f'Relative improvement of {sel_method_name}', **axis_font)
        else:
            ax.set_ylabel(f'Relative mean log {latex_metric_name}', **axis_font)
        ax.set_xlabel(f'mean log RMSE - mean log MAE', **axis_font)

    x_min, x_max = np.min(x_values), np.max(x_values)
    x_min = 0.0  # extend to 0.0
    ax.plot([x_min, x_max], [0.0, 0.0], 'k--', label=r'\textsc{Random}')

    # see https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    import scipy.stats
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_values, y_values)
    ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], '-', color='#888888',
            label='Linear Regression fit')
    print(f'plot_skewness_ax: R^2 = {r_value**2:g}')


def plot_error_variation(results: ExperimentResults, filename: Union[str, Path], metric_name: str, alg_name: str,
                         use_relative_improvement: bool = False):
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))

    plot_skewness_ax(ax, results, metric_name=metric_name, alg_name=alg_name,
                     use_relative_improvement=use_relative_improvement)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.02))
    # plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close(fig)


# ----- not needed plots -----


def plot_eff_dim(results: ExperimentResults, filename: Union[str, Path], pattern: str, metric_name: str):
    # plots, for every data set, error of grad_rp-512 to ll vs difference in effective dimensions (average),
    # for different selection methods (esp. maxdist, maxdet, lcmd, fw),
    all_alg_names = [pattern.replace('*', 'll'), pattern.replace('*', 'grad_rp-512')]
    all_task_names = {task_name for alg_name in all_alg_names for task_name in results.results_dict[alg_name].keys()}

    eff_dims = results.get_avg_al_stats('eff_dim')
    last_errors = results.get_last_errors(metric_name)
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in all_task_names})

    fig, axs = plt.subplots(figsize=(5, 5))

    for name_task in ds_names:
        log_means_ll = []
        log_means_grad_rp = []
        for key in last_errors.results_dict[all_alg_names[0]]:
            if key.startswith(name_task) and int(key.split('_')[-1].split('x')[0]) == 256:
                log_means_ll = [np.mean(last_errors.results_dict[all_alg_names[0]][key]), np.mean(eff_dims.results_dict[all_alg_names[0]][key])]
                log_means_grad_rp = [np.mean(last_errors.results_dict[all_alg_names[1]][key]), np.mean(eff_dims.results_dict[all_alg_names[1]][key])]

        log_means_ll = np.asarray(log_means_ll)
        log_means_grad_rp = np.asarray(log_means_grad_rp)

        axs.plot(log_means_grad_rp[1] - log_means_ll[1], log_means_ll[0] - log_means_grad_rp[0], 'o', label=escape(name_task))

    axs.legend()
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def plot_learning_curves_v2(results: ExperimentResults, filename: Union[str, Path], metric_name: str):
    # plot averaged learning curve for all tasks
    all_alg_names = list(results.results_dict.keys())
    all_task_names = {task_name for alg_name in all_alg_names for task_name in results.results_dict[alg_name].keys()}
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in all_task_names})
    algs_with_all_tasks = [alg_name for alg_name in all_alg_names
                           if set(results.results_dict[alg_name].keys()) == all_task_names]
    alg_names = algs_with_all_tasks + ['NN_lcmd-tp_ll', 'NN_kmeanspp-tp_ll', 'NN_maxdist-tp_grad_rp-512', 'NN_random']

    fig, axs = plt.subplots(figsize=(5, 5))
    plt.xscale('log')

    learning_curves = results.get_learning_curves(metric_name)

    n_train = np.asarray([256 * (i + 1) for i in range(17)])
    alg_results = {alg_name: np.mean([learning_curves.results_dict[alg_name][ds_name + '_256x16']
                            for ds_name in ds_names], axis=0) for alg_name in alg_names}

    for alg_name in alg_names:
        axs.plot(n_train, alg_results[alg_name] - alg_results['NN_random'], '--o', label=escape(alg_name))

    axs.legend()
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)


def alt_plot_eff_dim(results: ExperimentResults, filename: Union[str, Path], pattern: str, metric_name: str):
    # plots, for every data set, error of grad_rp-512 to ll vs difference in effective dimensions (average),
    # for different selection methods (esp. maxdist, maxdet, lcmd, fw),
    all_alg_names = [pattern.replace('*', 'll'), pattern.replace('*', 'grad_rp-512'), 'NN_random']
    all_task_names = {task_name for alg_name in all_alg_names for task_name in results.results_dict[alg_name].keys()}

    eff_dims = results.get_avg_al_stats('eff_dim')
    last_errors = results.get_last_errors(metric_name)
    ds_names = list({'_'.join(task_name.split('_')[:-1]) for task_name in all_task_names})

    fig, axs = plt.subplots(figsize=(5, 5))

    for name_task in ds_names:
        log_means_ll = []
        log_means_grad_rp = []
        log_means_random = []
        for key in last_errors.results_dict[all_alg_names[0]]:
            if key.startswith(name_task) and int(key.split('_')[-1].split('x')[0]) == 256:
                log_means_ll = [np.mean(last_errors.results_dict[all_alg_names[0]][key]),
                                np.mean(eff_dims.results_dict[all_alg_names[0]][key])]
                log_means_grad_rp = [np.mean(last_errors.results_dict[all_alg_names[1]][key]),
                                     np.mean(eff_dims.results_dict[all_alg_names[1]][key])]
                log_means_random = [np.mean(last_errors.results_dict[all_alg_names[2]][key]),
                                np.mean(eff_dims.results_dict[all_alg_names[2]][key])]

        log_means_ll = np.asarray(log_means_ll)
        log_means_grad_rp = np.asarray(log_means_grad_rp)
        log_means_random = np.asarray(log_means_random)

        axs.plot(log_means_grad_rp[1] - log_means_ll[1],
                 (log_means_ll[0] - log_means_grad_rp[0]) / (abs(log_means_grad_rp[0] - log_means_random[0])
                                                             + abs(log_means_ll[0] - log_means_random[0])),
                 'o', label=escape(name_task))

    axs.legend()
    plt.tight_layout()
    plot_name = Path(custom_paths.get_plots_path()) / results.exp_name / filename
    utils.ensureDir(plot_name)
    plt.savefig(plot_name)
    plt.close(fig)
