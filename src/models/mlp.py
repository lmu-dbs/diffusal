import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, Dropout, LogSoftmax
from torch_geometric.data import Data


class MLP(Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 hidden_size: int = 16,
                 dropout: float = 0.5,
                 activation=ReLU,
                 init_fn=None,
                 lr=0.005,
                 weight_decay=1e-4
                 ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.init_fn = init_fn

        self.transform = Sequential(Dropout(dropout),
                                    Linear(in_features, hidden_size),
                                    activation(),
                                    )
        self.predict = Sequential(Dropout(dropout),
                                  Linear(hidden_size, num_classes)
                                  )

        self.reset_parameters()

    def forward(self,
                data: Data,
                ) -> Tensor:
        x = data.x

        x = self.transform(x)
        predictions = self.predict(x)

        return predictions

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                if self.init_fn is not None:
                    self.init_fn(module.weight)
                    module.bias.data.fill_(0.01)
                else:
                    module.reset_parameters()


class MLPCommittee(Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 hidden_size: int = 16,
                 dropout: float = 0.5,
                 activation=ReLU,
                 init_fn=None,
                 lr=0.005,
                 weight_decay=1e-4
                 ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self.m_1 = Sequential(Dropout(dropout),
                              Linear(in_features, hidden_size),
                              activation(),
                              )
        self.m_2 = Sequential(Dropout(dropout),
                              Linear(in_features, hidden_size),
                              activation(),
                              )
        self.m_3 = Sequential(Dropout(dropout),
                              Linear(in_features, hidden_size),
                              activation(),
                              )

        self.predict = Sequential(Dropout(dropout),
                                  Linear(hidden_size, num_classes)
                                  )

        self.reset_parameters()

    def forward(self,
                data: Data,
                ) -> Tensor:
        x = data.x

        x_1 = self.m_1(x.clone())
        x_2 = self.m_2(x.clone())
        x_3 = self.m_3(x)
        combined = x_1 + x_2 + x_3

        return self.predict(combined)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()
