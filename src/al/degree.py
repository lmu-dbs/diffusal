import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utilities.data_utils import DataWrapper
from .activelearner import ActiveLearner, SelectionType


class DegreeSampling(ActiveLearner):
    selection_type: SelectionType = SelectionType.ZEROSHOT

    def __init__(self, datawrapper: DataWrapper, model_params, clf='mlp', **kwargs):
        super().__init__(datawrapper, model_params, clf=clf, **kwargs)

    def query(self, n, **kwargs) -> Tensor:
        graph = to_networkx(self.datawrapper.data.cpu())
        centrality = nx.centrality.degree_centrality(graph)
        centrality = torch.tensor(list(centrality.values()))

        # make values negative to have highest values first
        centrality = -1 * centrality
        values, indices = centrality.sort()
        mask = self.get_unlabeled_train_mask()
        mask = mask[indices]
        indices = indices[mask]

        return indices[:n]