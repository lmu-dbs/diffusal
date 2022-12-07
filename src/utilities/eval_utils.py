from typing import Tuple

import mlflow
import numpy
import pandas


def export_from_mlflow(mlflow_uri: str,
                       mlflow_experiment_name: str,
                       metrics: Tuple[str, ...],
                       ) -> pandas.DataFrame:
    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment by ID
    experiment = client.get_experiment_by_name(name=mlflow_experiment_name)
    experiment_id = experiment.experiment_id

    # Load parameters and metrics
    results_df = []
    for run in client.search_runs(experiment_ids=[experiment_id]):
        run_id = run.info.run_id

        data = run.data.params
        data.update({key: run.data.metrics[key] for key in run.data.metrics.keys() if key in metrics})
        data.update({'exp_name': mlflow_experiment_name})
        run_df = pandas.DataFrame(data=data,
                                  index=[run_id],
                                  )

        results_df += [run_df]

    results_df = pandas.concat(results_df,
                               sort=True,
                               )

    return results_df

def export_from_mlflow_with_sd(mlflow_uri: str,
                       mlflow_experiment_name: str,
                       metrics: Tuple[str, ...],
                       ) -> pandas.DataFrame:
    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment by ID
    experiment = client.get_experiment_by_name(name=mlflow_experiment_name)
    experiment_id = experiment.experiment_id

    # Load parameters and metrics
    results_df = []
    for run in client.search_runs(experiment_ids=[experiment_id]):
        run_id = run.info.run_id

        data = run.data.params
        data.update({key: run.data.metrics[key] for key in run.data.metrics.keys() if key in metrics})

        accs = []
        for metric in client.get_metric_history(run_id, 'test_acc'):
            accs.append(metric.value)
        data.update({'sd_test_acc': numpy.std(accs)})

        run_df = pandas.DataFrame(data=data,
                                  index=[run_id],
                                  )
        results_df += [run_df]

    results_df = pandas.concat(results_df,
                               sort=True,
                               )

    return results_df
