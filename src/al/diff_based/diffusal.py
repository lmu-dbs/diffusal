import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from .strategy_diff import DiffusionStrategy
from al.activelearner import SelectionType
from utilities.data_utils import DataWrapper


class DiffusAL(DiffusionStrategy):
    selection_type: SelectionType = SelectionType.ITERATIVE_SEQ

    def __init__(self, datawrapper: DataWrapper, model_params, clf='mlp', **kwargs):
        super().__init__(datawrapper, model_params, clf, **kwargs)

        if 'importance' in kwargs:
            importance = kwargs['importance']
        else:
            sparse = torch.sparse_coo_tensor(datawrapper.diffused.edge_index, datawrapper.diffused.edge_attr)
            sparse = sparse.cpu()
            importance = torch.sparse.sum(sparse, 0).to_dense()
        self.diffused_features = datawrapper.diffused_features
        importance = importance.to(self.clf.device)
        importance = importance / importance.max()
        self.importance = importance

        self.cluster_masks = None

    def initialize_labeled_pool(self,
                                num_train: int,
                                seed: int = 0,
                                num_val: int = 500,
                                num_test: int = 1000,
                                ):

        super().initialize_labeled_pool(
            num_train=0, seed=seed, num_test=num_test, num_val=num_val)

        unl_mask = self.get_unlabeled_train_indices()

        kmeans = KMeans(n_clusters=num_train, random_state=seed)
        X = self.datawrapper.diffused_features.x.cpu()
        clustering = kmeans.fit(X[unl_mask])
        labels = clustering.labels_
        cluster_centers = clustering.cluster_centers_
        self.cluster_masks = []
        for c in range(num_train):
            mask = labels == c
            self.cluster_masks.append(torch.from_numpy(mask).to(self.clf.device))

        closest, _ = pairwise_distances_argmin_min(cluster_centers, X[unl_mask])
        print(closest.shape)
        print(cluster_centers.shape)
        chosen = unl_mask[list(closest)]
        print(chosen)
        self.train_data.train_mask[chosen] = True

    def query(self, n: int, **kwargs) -> torch.Tensor:
        importance = self.get_importance_score()
        uncertainty = self.get_uncertainty_score()
        diversity = self.get_diversity_score()

        scores = importance * uncertainty * diversity
        values, indices = scores.sort()
        indices = indices.flip(0)

        mask = self.get_unlabeled_train_mask()
        mask = mask[indices]
        indices = indices[mask]
        return indices[:n]

    def get_importance_score(self):
        return self.importance

    def get_diversity_score(self):
        diversity = torch.zeros_like(self.importance)
        for cluster_mask in self.cluster_masks:
            train_cluster_mask = self.train_data.train_mask & cluster_mask
            num_div = train_cluster_mask.sum()
            fraction_div = num_div / self.train_data.train_mask.sum()
            div_score = 1 - fraction_div
            diversity[cluster_mask] = div_score

        return diversity

    def get_uncertainty_score(self):
        probs = self.predict_probabilities()
        entropy = -(probs * probs.log()).sum(dim=1)
        uncertainty = entropy / entropy.max()
        return uncertainty

    def get_embeddings(self):
        self.clf.model.eval()
        with torch.no_grad():
            return self.clf.model(self.train_data)