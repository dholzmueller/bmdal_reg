import sys
from .evaluation.plotting import *
from .evaluation.visualize_lcmd import create_lcmd_plots


def plot_all(results: ExperimentResults, alg_names: List[str], with_batch_size_plots: bool = True):
    selected_results = results.filter_alg_names(alg_names)
    literature_results = results.filter_alg_names(['NN_random', 'NN_maxdiag_ll_train', 'NN_maxdet-p_ll_train',
                                                   'NN_bait-fb-p_ll_train',
                                                   'NN_fw-p_ll_acs-rf-hyper-512', 'NN_maxdist-tp_ll',
                                                   'NN_kmeanspp-p_ll_train', 'NN_lcmd-tp_grad_rp-512'])
    literature_names = ['No BMAL', 'BALD', 'BatchBALD', 'BAIT', 'ACS-FW', 'Core-Set / FF-Active', 'BADGE', 'Ours']

    print('Generating tables...')
    save_latex_table_all_algs(results, 'table_all_algs.txt')
    save_latex_table_data_sets(selected_results, 'table_data_sets.txt')
    save_latex_table_data_sets(selected_results, 'table_data_sets_lasterror.txt', use_last_error=True)
    save_latex_table_data_sets(selected_results, 'table_data_sets_nolog.txt', use_log=False)
    save_latex_table_data_sets(selected_results, 'table_data_sets_nolog_lasterror.txt', use_log=False,
                               use_last_error=True)

    print('Creating learning curve plots...')
    plot_learning_curves_metrics_subplots(results=selected_results, filename='learning_curves_metrics.pdf')
    plot_multiple_learning_curves(results=selected_results, filename='learning_curves_rmse_maxe.pdf',
                                  metric_names=['rmse', 'maxe'])
    plot_multiple_learning_curves(results=selected_results, filename='learning_curves_q95_q99.pdf',
                                  metric_names=['q95', 'q99'])
    for metric_name in metric_names:
        plot_learning_curves(results=selected_results, filename=f'learning_curves_{metric_name}.pdf',
                             metric_name=metric_name)
        plot_learning_curves(results=literature_results, filename=f'learning_curves_literature_{metric_name}.pdf',
                             metric_name=metric_name, labels=literature_names, figsize=(6, 5))
        plot_learning_curves(results=literature_results, filename=f'learning_curves_literature_wide_{metric_name}.pdf',
                             metric_name=metric_name, labels=literature_names, figsize=(6, 3.5))

    print('Creating individual learning curve plots with subplots...')
    for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
        plot_learning_curves_individual_subplots(results=selected_results,
                                                 filename=f'learning_curves_individual_{metric_name}.pdf',
                                                 metric_name=metric_name)
    print('Creating error variation plots...')
    plot_error_variation(results, 'skewness_ri_lcmd-tp_grad_rp-512.pdf', metric_name='rmse', alg_name='NN_lcmd-tp_grad_rp-512',
                         use_relative_improvement=True)

    print('Creating correlation plots...')
    for metric_name in metric_names:
        plot_correlation_between_methods(results=selected_results,
                                         filename=f'correlation_between_methods_{metric_name}.pdf',
                                         metric_name=metric_name)
    print('Creating individual learning curve plots...')
    for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
        plot_learning_curves_individual(results=selected_results, metric_name=metric_name)

    if with_batch_size_plots:
        # batch size plots
        print('Creating batch size plots...')
        plot_batch_sizes_metrics_subplots(results=selected_results, filename='batch_sizes_metrics.pdf')
        plot_multiple_batch_sizes(results=selected_results, filename='batch_sizes_rmse_maxe.pdf',
                                  metric_names=['rmse', 'maxe'])
        plot_multiple_batch_sizes(results=selected_results, filename='batch_sizes_q95_q99.pdf',
                                  metric_names=['q95', 'q99'])
        for metric_name in metric_names:
            plot_batch_sizes(results=selected_results, filename=f'batch_sizes_{metric_name}.pdf',
                             metric_name=metric_name, figsize=(5, 5))
            plot_batch_sizes(results=selected_results, filename=f'batch_sizes_wide_{metric_name}.pdf',
                             metric_name=metric_name, figsize=(6, 3.5))
        print('Creating individual batch size plots with subplots...')
        for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
            plot_batch_sizes_individual_subplots(results=selected_results,
                                                 filename=f'batch_sizes_individual_{metric_name}.pdf',
                                                 metric_name=metric_name)
        print('Creating individual batch size plots...')
        for metric_name in ['mae', 'rmse', 'q95', 'q99', 'maxe']:
            plot_batch_sizes_individual(results=selected_results, metric_name=metric_name)


if __name__ == '__main__':
    metric_names = ['mae', 'rmse', 'q95', 'q99', 'maxe']
    if len(sys.argv) > 1:
        exp_names = [sys.argv[1]]
    else:
        available_names = utils.getSubfolderNames(custom_paths.get_results_path())
        exp_names = [name for name in available_names if name in ['relu', 'silu']]

    for exp_name in exp_names:
        print(f'----- Running evaluation for {exp_name} experiments -----')
        print('Loading experiment results...')
        results = ExperimentResults.load(exp_name)
        print('Loaded experiment results')
        print_avg_results(results)
        # print_all_task_results(results)
        print('Analyzing results')
        results.analyze_errors()
        results.analyze_eff_dims()

        if exp_name == 'relu':
            # selected algs for ReLU (best ones in terms of RMSE after ignoring slow ones, see table in the paper)
            alg_names_relu = ['NN_random', 'NN_maxdiag_grad_rp-512_acs-rf-512', 'NN_maxdet-p_grad_rp-512_train',
                              'NN_bait-f-p_grad_rp-512_train',
                              'NN_fw-p_grad_rp-512_acs-rf-hyper-512', 'NN_maxdist-p_grad_rp-512_train',
                              'NN_kmeanspp-p_grad_rp-512_acs-rf-512',
                              'NN_lcmd-tp_grad_rp-512']
            plot_all(results, alg_names=alg_names_relu)
        elif exp_name == 'silu':
            # selected algs for SiLU
            alg_names_silu = ['NN_random', 'NN_maxdiag_grad_rp-512_train', 'NN_maxdet-p_grad_rp-512_train',
                              'NN_bait-f-p_grad_rp-512_train',
                              'NN_fw-p_grad_rp-512_acs-rf-hyper-512', 'NN_maxdist-tp_grad_rp-512',
                              'NN_kmeanspp-tp_grad_rp-512',
                              'NN_lcmd-tp_grad_rp-512']
            plot_all(results, alg_names=alg_names_silu, with_batch_size_plots=False)

        print('Finished plotting')
        print()

    print('Creating lcmd visualization...')
    create_lcmd_plots(n_train=1, n_pool=500, n_steps=20)
