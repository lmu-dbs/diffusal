from torch_geometric.data import Data

from al.activelearner import ActiveLearner
from utilities.data_utils import DataWrapper


class DiffusionStrategy(ActiveLearner):

    def __init__(self, datawrapper: DataWrapper, model_params, clf='mlp', **kwargs):
        super().__init__(datawrapper, model_params, clf=clf, **kwargs)
