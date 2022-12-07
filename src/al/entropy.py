from torch import Tensor
from torch_geometric.data import Data

from utilities.data_utils import DataWrapper
from .activelearner import ActiveLearner, SelectionType


class EntropySampling(ActiveLearner):
    selection_type: SelectionType = SelectionType.ITERATIVE

    def __init__(self, datawrapper: DataWrapper, model_params, clf='mlp', **kwargs):
        super().__init__(datawrapper, model_params, clf=clf, **kwargs)
        self.train_data = self.train_data.to(self.clf.device)

    def query(self, n: int, **kwargs) -> Tensor:
        probs = self.clf.predict_probs(self.train_data)
        # keep values negative, so highest entropy comes first when sorting
        entropy = (probs * probs.log()).sum(1)
        values, indices = entropy.sort()

        mask = self.get_unlabeled_train_mask()
        # "sort" mask, based on indices
        mask = mask[indices]
        indices = indices[mask]
        return indices[:n]
