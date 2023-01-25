from bmdal_reg.run_experiments import get_relu_configs, run_experiments, get_silu_configs


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
