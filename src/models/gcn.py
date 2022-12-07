from torch import relu, Tensor, log_softmax
from torch.nn import Module
from torch.nn.functional import dropout
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 hidden_size: int = 16,
                 dropout: float = 0.5,
                 lr=0.005,
                 weight_decay=1e-4
                 ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels=in_features,
                             out_channels=hidden_size,
                             normalize=True,
                             add_self_loops=True,
                             cached=True,
                             )
        self.conv2 = GCNConv(in_channels=hidden_size,
                             out_channels=num_classes,
                             normalize=True,
                             add_self_loops=True,
                             cached=True,
                             )

        self.reset_parameters()

    def forward(self,
                data: Data,
                ) -> Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x,
                       edge_index=edge_index,
                       edge_weight=edge_weight,
                       )
        x = relu(x)

        x = dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x,
                       edge_index=edge_index,
                       edge_weight=edge_weight,
                       )

        return x

    def get_embedding(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x,
                       edge_index=edge_index,
                       edge_weight=edge_weight,
                       )
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
