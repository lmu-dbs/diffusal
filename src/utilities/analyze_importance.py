from os.path import join
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, sparse_coo_tensor
from torch_geometric.data import Data

from utilities.data_utils import load_labels



def calculate_importance(edge_index: Tensor, edge_weight: Tensor) -> Tensor:
    sparse = sparse_coo_tensor(edge_index, edge_weight)
    importance = torch.sparse.sum(sparse, 0).to_dense()
    return importance


def save_importance(importance: Tensor, dataset_name: str, data_root: str) -> None:
    path = join(data_root, 'cached', dataset_name.lower(), 'importance.pt')
    torch.save(importance, path)


def load_importance(dataset_name: str, data_root: str) -> Tensor:
    path = join(data_root, 'cached', dataset_name.lower(), 'importance.pt')
    return torch.load(path)


def load_importance_sorted(dataset_name: str, data_root: str) -> Tuple[Tensor]:
    path = join(data_root, 'cached', dataset_name, 'importance.pt')
    importance = torch.load(path)
    sorted_importance = importance.sort()

    values = sorted_importance[0]
    indices = sorted_importance[1]

    values = values.flip(0)
    indices = indices.flip(0)

    return values, indices


def get_class_distribution(indices: Tensor,
                            y: Tensor,
                            budget: Union[int, List[int]],
                            ) -> Tensor:
    """
    For a given budget or list of budgets, calculate the frequencies
    of each class among the most important nodes.
    """

    num_classes = y.unique().numel()
    def _get_dist(budget):
        dist = torch.zeros(num_classes, dtype=torch.int32)
        top_b_nodes = indices[:budget]
        top_b_classes = y[top_b_nodes]

        for c in range(num_classes):
            dist[c] = top_b_classes[top_b_classes == c].shape[0]
        return dist
    
    if type(budget) is int:
        return _get_dist(budget)
    else:
        class_distributions = torch.zeros(len(budget), num_classes, dtype=torch.int32)
        for b in range(len(budget)):
            class_distributions[b] = _get_dist(budget[b])            
        return class_distributions


def get_relative_distribution(importance_indices: Tensor,
                                        y: Tensor,
                                        budget: Union[int, List[int]],
                                        ) -> Tensor:
    """
    Calculates the class distribution among the most important nodes
    relative to the budget (fraction of nodes for a given class in the set)
    """

    dist = get_class_distribution(importance_indices, y, budget)
    return (dist.T / torch.tensor(budget)).T

def plot_distribution(distribution: Tensor, budget: Union[int, List[int]], title: str) -> None:
    if type(budget) is int:
        labels = [str(budget)]
        df = pd.DataFrame(np.expand_dims(distribution.numpy(), axis=0)) 
    else:
        labels = list(map(lambda x: str(x), budget))
        df = pd.DataFrame(distribution.numpy())

    df.index = labels
    df.plot(kind='bar', stacked=True, title=title)


def plot_tsne_distribution(tsne: Tensor, indices: Tensor, y: Tensor, budget: Union[int, List[int]], title: str) -> None:
    color_map = ['b','g','r','c','m','y','k','w']
    fig, ax = plt.subplots(4,3, figsize=(10,10))
    fig.suptitle(title)
    fig.tight_layout()

    for i, b in enumerate(budget):
        plot = ax[i//3, i%3]
        for c in range(y.unique().numel()):
            filtered = tsne[indices[:b]]
            filtered = filtered[y[indices[:b]]==c]
            plot.set_title(str(b))
            plot.plot(filtered[:, 0], filtered[:, 1], f'{color_map[c]}o', markersize=5)
    plt.show()
