import torch
from torch import Tensor
from torch_geometric.data import Data

from utilities.data_utils import DataWrapper
from .activelearner import ActiveLearner, SelectionType


class RandomSampling(ActiveLearner):
    selection_type: SelectionType = SelectionType.ZEROSHOT

    def __init__(self, datawrapper: DataWrapper, model_params, clf='mlp', **kwargs):
        super().__init__(datawrapper, model_params, clf=clf, **kwargs)

    def query(self, n: int, **kwargs) -> Tensor:
        unlabeled = self.get_unlabeled_train_indices()
        random = torch.randperm(unlabeled.shape[0])
        return unlabeled[random][:n]
