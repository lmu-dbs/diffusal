import os
import mlflow
import numpy as np
import pandas as pd
import torch

from al import get_strategy_class
from utilities.experiment_utils import get_dataset, set_seed
from utilities.data_utils import flatten_dict, DataWrapper


if __name__ == '__main__':
    mlflow.set_tracking_uri('../mlruns')
    
    # datasets to evaluate
    datasets = [
        'Cora',
        'Citeseer',
        'Pubmed',
        'CS',
        'Physics'
    ]

    # random seed used to initialize different train/test/val sets
    seeds = [1,2,3,4,5,6,7,8,9,10]

    # DiffusAL
    DiffusAL = get_strategy_class("diffusal")

    # ablations
    DiffNoUnc = get_strategy_class("diff_nounc")
    DiffNoImp = get_strategy_class("diff_noimp")
    DiffNoDiv = get_strategy_class("diff_nodiv")

    strat_params_all = [
        {"clf": "qbc"},  # DiffusAL CLF
        {"clf": "mlp"},
        {"clf": "gcn"}
    ]

    torch.cuda.set_device(0)

    # the baseline strategies to evaluate
    baselines = [
        DiffusAL,
        DiffNoImp,
        DiffNoUnc,
        DiffNoDiv,
    ]

    for dataset in datasets:
        for strat_params in strat_params_all:
            mlflow.set_experiment(f'DiffusAL_{dataset}')
            epsilon = 1e-5

            data_wrapper = DataWrapper(
                data=get_dataset(dataset_name=dataset,diffusion=False,diffused_features=False, add_self_loops=False, adj_normalization=None),
                diffused=get_dataset(dataset_name=dataset, diffusion=True, diffused_features=False, epsilon=epsilon),
                diffused_features=get_dataset(dataset_name=dataset, diffusion=True, diffused_features=True, epsilon=epsilon)
            )

            num_classes = data_wrapper.data.y.unique().numel()
            m_params = {
                "num_classes": num_classes,
                "hidden_size": 16,
                "dropout": 0.5,
                "lr": 0.01,
                "weight_decay": 1e-4
            }

            qs = 2 * num_classes
            rounds = 10
            budgets = [qs*i for i in range(1,rounds+1)]

            for baseline in baselines:
                strat_params["dataset_name"] = dataset

                finally_labeled = budgets[-1]
                params = {
                    'dataset': dataset,
                    'rounds': rounds,
                    'qs': qs,
                    'baseline': baseline.__name__,
                    'init_size': budgets[0],
                    'labeled': finally_labeled,
                    'strategy_params': strat_params,
                    'model_params': m_params
                }
                print(params, end='\r')

                with mlflow.start_run() as run:
                    mlflow.log_params(flatten_dict(params))

                    _accs_all = []

                    for seed in seeds:
                        set_seed(seed)
                        _accs_seed = []

                        strat = baseline(data_wrapper, model_params=m_params, **strat_params)

                        for i, budget in enumerate(budgets):
                            # select indices according to strategy and update labeled pool
                            strat.loop_step(budget, i, qs, seed)

                            print(f"\t Round {i} | QS: {qs} | Currently Labeled: {int(sum(strat.train_data.train_mask))} | Round Budget: {budget}")
                            # fresh training for fair evaluation, with validation set
                            strat.train(max_epochs=1000, patience=25, reset_params=True)
                            test_acc = strat.get_test_accuracy()
                            mlflow.log_metric(f'test_acc_{seed}', test_acc, step=int(sum(strat.train_data.train_mask)))
                            _accs_seed.append(test_acc)

                            # intermediate training without validation -> next acquisition batch should build upon model without val set training
                            strat.train_intermediate(num_epochs=200, reset_params=True)

                        _accs_all.append(_accs_seed)

                    avg_accs = np.mean(np.asarray(_accs_all), axis=0)
                    std_accs = np.std(np.asarray(_accs_all), axis=0)

                    for i, val in enumerate(avg_accs):
                        step = budgets[i]
                        # accuracy
                        mlflow.log_metric('mean_test_acc', val, step=step)
                        mlflow.log_metric('std_test_acc', std_accs[i], step=step)
