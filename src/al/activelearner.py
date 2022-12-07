import torch
from torch import Tensor
from torch_geometric.data import Data

from models.classifier import TorchClassifier
from models.mlp import MLP, MLPCommittee, LinearMLP
from models.pushnet import PushNetTPP, PushNetPTP
from models.gcn import GCN
from utilities.data_utils import DataWrapper, load_dgi_features
from enum import Enum
from models.distance import DistCLF


class SelectionType(Enum):
    ITERATIVE = "iterative"
    ITERATIVE_SEQ = "iterative_sequential"
    ONESHOT = "oneshot"
    ZEROSHOT = "zeroshot"


class ActiveLearner:
    selection_type: SelectionType = SelectionType.ITERATIVE
    sequential_selection = False

    def __init__(self, datawrapper: DataWrapper, model_params, clf='mlp', **kwargs):
        self.datawrapper = datawrapper
        self.train_data: Data
        if "dataset_name" in kwargs:
            self.dataset_name = kwargs["dataset_name"]

        self.clf: TorchClassifier
        self.set_clf(m_params=model_params, clf=clf)
        print(type(self.clf.model))
        print(self.clf.model)
        self.train_indices = None

    def query(self, n: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def loop_step(self, budget, acq_round, qs, seed) -> Tensor:
        """
        This method performs one step in the current acquisition loop depending on the active learning selection type.
        budget: total number of labeled instances
        acq_round: current sampling iteration
        qs: query size, number of labels that should be queried on top in this acquisition round
        seed: seed
        """
        # iterative: select query_size instances on top of already labeled instances, assumes that model is trained on previously labeled dataset
        if self.selection_type == SelectionType.ITERATIVE:
            if acq_round == 0:
                self.initialize_labeled_pool(num_train=budget, seed=seed)
            else:
                self.update(self.query(n=qs, acq_round=acq_round))
        # iterative_seq: special case of iterative, additive training within acquisition step
        elif self.selection_type == SelectionType.ITERATIVE_SEQ:
            if acq_round == 0:
                self.initialize_labeled_pool(num_train=budget, seed=seed)
            else:
                queried = 0
                while queried < qs:
                    self.update(self.query(1))
                    # incremental training within one acquisition step
                    self.train_intermediate(num_epochs=1, reset_params=False)
                    queried += 1
        # zeroshot: select <budget> instances from scratch, totally model agnostic
        elif self.selection_type == SelectionType.ZEROSHOT:
            self.initialize_labeled_pool(num_train=0, seed=seed)
            # no intermediate training, always start from scratch for each budget
            self.update(self.query(budget))
        # oneshot: select <budget> instances on top of initially labeled instances, assumes that model is trained only on initial samples
        elif self.selection_type == SelectionType.ONESHOT:
            self.initialize_labeled_pool(num_train=qs, seed=seed)
            if acq_round > 0:
                self.train_intermediate(num_epochs=200, reset_params=True)
                self.update(self.query(budget))

    def train(self, **kwargs):
        self.clf.train(data=self.train_data, **kwargs)

    def train_intermediate(self, num_epochs=300, reset_params=False):
        self.clf.train_intermediate(data=self.train_data, num_epochs=num_epochs, reset_params=reset_params)

    def predict_probabilities(self):
        return self.clf.predict_probs(self.train_data)

    def get_test_accuracy(self):
        return self.clf.best_accuracy

    def initialize_labeled_pool(
        self,
        num_train: int = 5,
        seed: int = 0,
        num_val: int = 500,
        num_test: int = 1000
    ):
        # initialize new train, val and test masks
        self._zero_masks()

        # fix random seed for consistent train and test set
        torch.manual_seed(seed)

        # find new indices according to the specified amounts of nodes
        indices = torch.randperm(self.train_data.num_nodes)

        test_indices = indices[-num_test:]
        self.train_data.test_mask[test_indices] = True

        val_indices = indices[-(num_val + num_test):-num_test]
        self.train_data.val_mask[val_indices] = True

        self.train_indices = indices[:-(num_val + num_test)]
        self.train_data.train_mask[self.train_indices[:num_train]] = True

    def update(self, indices: Tensor):
        self.train_data.train_mask[indices] = True

    def _zero_masks(self):
        self.train_data.train_mask = torch.zeros(
            self.train_data.num_nodes, dtype=torch.bool)
        self.train_data.val_mask = torch.zeros(
            self.train_data.num_nodes, dtype=torch.bool)
        self.train_data.test_mask = torch.zeros(
            self.train_data.num_nodes, dtype=torch.bool)

    def get_unlabeled_train_indices(self):
        return (~(self.train_data.train_mask + self.train_data.val_mask + self.train_data.test_mask)) \
            .nonzero(as_tuple=False).view(-1)

    def get_labeled_train_indices(self):
        return self.train_data.train_mask.nonzero(as_tuple=False).view(-1)

    def get_unlabeled_train_mask(self):
        return ~(self.train_data.train_mask + self.train_data.val_mask + self.train_data.test_mask)

    def set_clf(self, m_params, clf='mlp'):
        if clf == 'mlp':
            data = self.datawrapper.diffused_features
            model = MLP(in_features=data.num_features, **m_params)
        elif clf == 'qbc':
            data = self.datawrapper.diffused_features
            model = MLPCommittee(in_features=data.num_features, **m_params)
        elif clf == "gcn":
            data = self.datawrapper.data
            model = GCN(in_features=data.num_features, **m_params)
        else:
            raise NotImplementedError('Possible classifiers are: RandomForest(rf), MLP(mlp) and MLPCommittee(qbc)')
        self.clf = TorchClassifier(model)
        self.train_data = data.to(self.clf.device)
