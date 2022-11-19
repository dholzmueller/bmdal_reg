from .bmdal.algorithms import BatchSelectorImpl, select_batch
from .models import *
from .data import *
from . import utils
from pathlib import Path


class ModelTrainer:
    def __init__(self, alg_name: str, create_model=create_tabular_model, n_models: int = 1, **config):
        self.alg_name = alg_name
        self.create_model = create_model
        self.config = config
        self.n_models = n_models

    def get_result_file_path(self, exp_name: str, task_name: str, split_id: int) -> Path:
        results_path = Path(custom_paths.get_results_path())
        result_file = results_path / exp_name / task_name / self.alg_name / str(split_id) / 'results.json'
        return result_file

    def __call__(self, task_split: TaskSplit, device: str, do_timing: bool = False):
        # results_path = Path(custom_paths.get_results_path())
        # result_file = results_path / task_split.task_name / self.alg_name / str(task_split.id) / 'results.json'
        al_device = 'cpu' if self.config.get('al_on_cpu', False) else device

        # can lead to imprecise distance calculations otherwise on devices with TF32
        torch.backends.cuda.matmul.allow_tf32 = False

        # if utils.existsFile(result_file) and not self.config.get('rerun', False):
        #     print(f'Results already exist for {self.alg_name} on split {task_split.id} of task {task_split.task_name}',
        #           flush=True)
        #     return

        print(f'Running {self.alg_name} on split {task_split.id} of task {task_split.task_name} on device {device}',
              flush=True)

        np.random.seed(task_split.id)
        torch.manual_seed(task_split.id)

        base_kernel = self.config.get('base_kernel', None)
        # task_split should contain train_idxs, valid_idxs, test_idxs, pool_idxs
        train_idxs = torch.as_tensor(task_split.train_idxs, dtype=torch.int64, device=device)
        valid_idxs = torch.as_tensor(task_split.valid_idxs, dtype=torch.int64, device=device)
        pool_idxs = torch.as_tensor(task_split.pool_idxs, dtype=torch.int64, device=device)
        test_idxs = torch.as_tensor(task_split.test_idxs, dtype=torch.int64, device=device)
        data = task_split.data.to(device)
        n_features = data.tensors['X'].shape[1]

        train_timer = utils.Timer()
        al_timer = utils.Timer()
        al_stats_list = []

        if base_kernel is None:
            # train on random dataset
            model = self.create_model(n_models=self.n_models, n_features=n_features, **self.config).to(device)
            n_train_missing = sum(task_split.al_batch_sizes)
            random_pool_idxs = pool_idxs[torch.randperm(len(task_split.pool_idxs), device=device)[:n_train_missing]]
            train_idxs = torch.cat([train_idxs, random_pool_idxs])
            train_timer.start()
            fit_model(model, data, self.n_models, train_idxs, valid_idxs, **self.config)
            train_timer.pause()
            results = [test_model(model, data, self.n_models, test_idxs)]
        else:
            results = []

            model = self.create_model(n_models=self.n_models, n_features=n_features, **self.config).to(device)
            train_timer.start()
            fit_model(model, data, self.n_models, train_idxs, valid_idxs, **self.config)
            train_timer.pause()
            results.append(test_model(model, data, self.n_models, test_idxs))

            for al_step, al_batch_size in enumerate(task_split.al_batch_sizes):
                print(f'Performing AL step {al_step+1}/{len(task_split.al_batch_sizes)} with n_train={len(train_idxs)}'
                      f', n_pool={len(pool_idxs)}, al_batch_size={al_batch_size}', flush=True)
                single_models = [model.get_single_model(i).to(al_device) for i in range(self.n_models)]
                X = TensorFeatureData(data.tensors['X'].to(al_device))
                feature_data = {'train': X[train_idxs],
                                'pool': X[pool_idxs]}
                y_train = data.tensors['y'][train_idxs].to(al_device)

                al_timer.start()
                new_idxs, al_stats = select_batch(models=single_models, data=feature_data, y_train=y_train,
                                                  use_cuda_synchronize=do_timing,
                                                  **utils.update_dict(self.config, {'batch_size': al_batch_size}))
                al_timer.pause()
                al_stats_list.append(al_stats)
                if str(device).startswith('cuda'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                new_idxs = new_idxs.to(device)
                # print(f'{len(new_idxs)=}')
                logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool, device=device)
                logical_new_idxs[new_idxs] = True
                train_idxs = torch.cat([train_idxs, pool_idxs[logical_new_idxs]], dim=-1)
                pool_idxs = pool_idxs[~logical_new_idxs]
                # print(f'{train_idxs.shape[-1]=}')
                # print(f'{pool_idxs.shape[-1]=}')

                model = self.create_model(n_models=self.n_models, n_features=n_features, **self.config).to(device)
                train_timer.start()
                fit_model(model, data, self.n_models, train_idxs, valid_idxs, **self.config)
                train_timer.pause()
                results.append(test_model(model, data, self.n_models, test_idxs))

        extended_config = utils.join_dicts(self.config, {'alg_name': self.alg_name, 'n_models': self.n_models})

        results = {'errors': {key: [r[key] for r in results] for key in ['mae', 'rmse', 'maxe', 'q95', 'q99']},
                   'train_times': train_timer.get_result_dict(),
                   'al_times': al_timer.get_result_dict(),
                   'al_stats': al_stats_list,
                   'config': extended_config}

        # if self.config.get('save', True):
        #     utils.serialize(result_file, results, use_json=True)

        print(f'Finished running {self.alg_name} on split {task_split.id} of task {task_split.task_name} on device {device}', flush=True)

        return results


def test_model(model, data, n_models, test_idxs):
    test_dl = ParallelDictDataLoader(data, test_idxs.expand(n_models, -1), batch_size=8192, shuffle=False,
                                     adjust_bs=False, drop_last=False)
    with torch.no_grad():
        model.eval()
        # for batch in test_dl:
        #     print(batch['y'].shape, batch['X'].shape, model(batch['X']).shape)
        errors = torch.cat([torch.abs(batch['y'] - model(batch['X'])) for batch in test_dl], dim=1).squeeze(-1)
    n_models = errors.shape[0]
    mae = errors.mean().item()
    rmse = (errors**2).mean().sqrt().item()
    maxe = torch.max(errors, dim=1)[0].mean().item()
    q95_errors = []
    q99_errors = []
    for i in range(n_models):
        errors_sorted, _ = torch.sort(errors[i])
        q95_errors.append(errors_sorted[int(0.95 * test_idxs.shape[-1])].item())
        q99_errors.append(errors_sorted[int(0.99 * test_idxs.shape[-1])].item())
    q95 = np.mean(q95_errors)
    q99 = np.mean(q99_errors)
    print(f'Test results: MAE={mae:g}, RMSE={rmse:g}, MAXE={maxe:g}, q95={q95:g}, q99={q99:g}')
    print('\n', flush=True)
    return {'mae': mae, 'rmse': rmse, 'maxe': maxe, 'q95': q95, 'q99': q99}


def fit_model(model, data, n_models, train_idxs, valid_idxs, n_epochs=256, batch_size=256, lr=3e-1, weight_decay=0.0,
              valid_batch_size=8192, **config):
    train_dl = ParallelDictDataLoader(data, train_idxs.expand(n_models, -1), batch_size=batch_size, shuffle=True,
                                      adjust_bs=False, drop_last=True)
    valid_dl = ParallelDictDataLoader(data, valid_idxs.expand(n_models, -1), batch_size=valid_batch_size, shuffle=False,
                                      adjust_bs=False, drop_last=False)
    n_steps = n_epochs * len(train_dl)
    best_valid_rmses = [np.Inf] * n_models
    best_model_params = [p.detach().clone() for p in model.parameters()]
    if config.get('opt_name', 'adam') == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        if weight_decay > 0.0:
            opt = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)
        else:
            opt = torch.optim.Adam(model.parameters())
    step = 0
    for i in range(n_epochs):
        # do one training epoch
        # grad_nonzeros = 0
        model.train()
        lr_sched = config.get('lr_sched', 'lin')
        for batch in train_dl:
            X, y = batch['X'], batch['y']
            y_pred = model(X)  # shape: n_models x batch_size x 1
            loss = ((y - y_pred)**2).mean(dim=-1).mean(dim=-1).sum()  # sum over n_models
            loss.backward()
            if lr_sched == 'lin':
                current_lr = lr * (1.0 - step / n_steps)
            elif lr_sched == 'hat':
                current_lr = lr * 2 * (0.5 - np.abs(0.5 - step/n_steps))
            elif lr_sched == 'warmup':
                peak_at = 0.1
                current_lr = lr * min((step/n_steps)/peak_at, (1-step/n_steps)/(1-peak_at))
            else:
                raise ValueError(f'Unknown lr sched "{lr_sched}"')
            for group in opt.param_groups:
                group['lr'] = current_lr
            opt.step()
            with torch.no_grad():
                for param in model.parameters():
                    # grad_nonzeros += torch.count_nonzero(param.grad).item()
                    param.grad = None

            step += 1

        # do one valid epoch
        valid_sses = torch.zeros(n_models, device=data.device)
        model.eval()
        with torch.no_grad():
            # linear_layers = [module for module in model.modules() if isinstance(module, ParallelLinearLayer)]
            # hooks = [ll.register_forward_hook(
            #     lambda layer, inp, out:
            #     print(f'dead neurons: {(out < 0).all(dim=0).all(dim=0).count_nonzero().item()}'))
            #     for ll in linear_layers]
            for batch in valid_dl:
                X, y = batch['X'], batch['y']
                valid_sses = valid_sses + ((y - model(X))**2).mean(dim=-1).sum(dim=-1)
            valid_rmses = torch.sqrt(valid_sses / len(valid_idxs)).detach().cpu().numpy()
            # for hook in hooks:
            #     hook.remove()

        # mean_param_norm = np.mean([p.norm().item() for p in model.parameters()])
        # first_param_mean_abs = list(model.parameters())[0].abs().mean().item()
        # print(f'Epoch {i+1}, Valid RMSEs: {valid_rmses}, first param mean abs: {first_param_mean_abs:g}, '
        #       f'grad nonzeros: {grad_nonzeros}')
        print('.', end='')
        for i in range(n_models):
            if valid_rmses[i] < best_valid_rmses[i]:
                best_valid_rmses[i] = valid_rmses[i]
                for p, best_p in zip(model.parameters(), best_model_params):
                    best_p[i] = p[i]

    print('', flush=True)

    with torch.no_grad():
        for p, best_p in zip(model.parameters(), best_model_params):
            p.set_(best_p)






