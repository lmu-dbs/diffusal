import os
from typing import Tuple, List, Dict

import networkx
import numpy
import torch.optim as optim
import scipy
import torch
from sklearn.preprocessing import normalize
from torch import Tensor
from torch_geometric.data import Dataset, Data, InMemoryDataset, download_url
from torch_geometric.datasets import Actor, Planetoid, Coauthor, Amazon, CitationFull, WebKB, WikipediaNetwork
from torch_geometric.utils import to_networkx, from_networkx, add_remaining_self_loops, is_undirected, to_undirected, \
    degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from models.dgi import DGI


def flatten_dict(d):
    """
    Function to transform a nested dictionary to a flattened dot notation dictionary.

    :param d: Dict
        The dictionary to flatten.

    :return: Dict
        The flattened dictionary.
    """

    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def get_preprocessed_path(dataset_name: str,
                          data_root: str,
                          sub_folder: str = None,
                          make_dirs: bool = True,
                          ) -> str:
    sub_path = os.path.join(data_root,
                            'preprocessed',
                            dataset_name,
                            )
    if sub_folder is not None:
        sub_path = os.path.join(sub_path, sub_folder)
    if not os.path.exists(sub_path):
        if make_dirs:
            os.makedirs(sub_path)
    return sub_path


def save_adj(edge_index: Tensor,
             edge_weight: Tensor,
             only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             dataset_name: str,
             data_root: str,
             ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    file_name = f'{dataset_name}'
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += f'_{adj_normalization}'

    file_name_index = file_name + '.edge_index'
    torch.save(edge_index,
               os.path.join(save_dir, file_name_index),
               )

    if edge_weight is not None:
        file_name_weight = file_name + '.edge_weight'
        torch.save(edge_weight,
                   os.path.join(save_dir, file_name_weight),
                   )


def save_ppr(edge_index: Tensor,
             edge_weight: Tensor,
             only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             alpha: float,
             epsilon: float,
             dataset_name: str,
             data_root: str,
             ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     )
    file_name = f'{dataset_name}'
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += f'_{adj_normalization}'
    file_name += '_alpha={:.2e}_epsilon={:.2e}'.format(alpha, epsilon)
    file_name_edge_index = file_name + '.edge_index'
    file_name_edge_weight = file_name + '.edge_weight'

    torch.save(edge_index,
               os.path.join(save_dir, file_name_edge_index),
               )
    torch.save(edge_weight,
               os.path.join(save_dir, file_name_edge_weight),
               )


def save_ppr_aggregated(edge_index: Tensor,
                        edge_weight: Tensor,
                        dataset_name: str,
                        data_root: str,
                        ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     )
    file_name = f'{dataset_name}_aggregated'
    file_name_edge_index = file_name + '.edge_index'
    file_name_edge_weight = file_name + '.edge_weight'

    torch.save(edge_index,
               os.path.join(save_dir, file_name_edge_index),
               )
    torch.save(edge_weight,
               os.path.join(save_dir, file_name_edge_weight),
               )


def save_ppr_role_descriptors(role_descriptors: Tensor,
                              only_largest_cc: bool,
                              add_self_loops: bool,
                              adj_normalization: str,
                              alpha: float,
                              epsilon: float,
                              dataset_name: str,
                              data_root: str,
                              ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     )
    file_name = f'{dataset_name}'
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += f'_{adj_normalization}'
    file_name += '_alpha={:.2e}_epsilon={:.2e}'.format(alpha, epsilon)
    file_name += '.role_descriptors'

    torch.save(role_descriptors,
               os.path.join(save_dir, file_name),
               )


def save_features(x: Tensor,
                  normalization: str,
                  dataset_name: str,
                  data_root: str,
                  ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    file_name = f'{dataset_name}'
    if normalization is not None:
        file_name += f'_{normalization}'
    file_name += '.x'

    torch.save(x,
               os.path.join(save_dir, file_name)
               )


def save_labels(y: Tensor,
                dataset_name: str,
                data_root: str,
                ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    file_name = f'{dataset_name}.y'

    torch.save(y,
               os.path.join(save_dir, file_name)
               )


def load_adj(only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             dataset_name: str,
             data_root: str,
             ) -> Tuple[Tensor, Tensor]:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    file_name = f'{dataset_name}'
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += f'_{adj_normalization}'

    file_name_index = file_name + '.edge_index'
    edge_index = torch.load(os.path.join(preprocessed_path, file_name_index))

    try:
        file_name_weight = file_name + '.edge_weight'
        edge_weight = torch.load(os.path.join(
            preprocessed_path, file_name_weight))
    except FileNotFoundError:
        edge_weight = None

    return edge_index, edge_weight


def load_ppr(only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             alpha: float,
             epsilon: float,
             dataset_name: str,
             data_root: str,
             ) -> Tuple[Tensor, Tensor]:
    ppr_path = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     make_dirs=False,
                                     )
    file_name = f'{dataset_name}'
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += f'_{adj_normalization}'
    file_name += '_alpha={:.2e}_epsilon={:.2e}'.format(alpha, epsilon)
    file_name_edge_index = file_name + '.edge_index'
    file_name_edge_weight = file_name + '.edge_weight'

    edge_index = torch.load(os.path.join(ppr_path, file_name_edge_index))
    edge_weight = torch.load(os.path.join(ppr_path, file_name_edge_weight))

    return edge_index, edge_weight


def load_ppr_aggregated(dataset_name: str,
                        data_root: str,
                        ) -> Tuple[Tensor, Tensor]:
    ppr_path = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     make_dirs=False,
                                     )
    file_name = f'{dataset_name}_aggregated'
    file_name_edge_index = file_name + '.edge_index'
    file_name_edge_weight = file_name + '.edge_weight'

    edge_index = torch.load(os.path.join(ppr_path, file_name_edge_index))
    edge_weight = torch.load(os.path.join(ppr_path, file_name_edge_weight))

    return edge_index, edge_weight


def load_features(normalization: str,
                  dataset_name: str,
                  data_root: str,
                  ) -> Tensor:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    file_name = f'{dataset_name}'
    if normalization is not None:
        file_name += f'_{normalization}'
    file_name += '.x'

    x = torch.load(os.path.join(preprocessed_path, file_name))
    return x


def load_labels(dataset_name: str,
                data_root: str,
                ) -> Tensor:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    file_name = f'{dataset_name}.y'

    y = torch.load(os.path.join(preprocessed_path, file_name))
    return y


def load_dataset_preprocessed(only_largest_cc: bool,
                              add_self_loops: bool,
                              adj_normalization: str,
                              x_normalization: str,
                              dataset_name: str,
                              data_root: str,
                              alpha: float = None,
                              epsilon: float = None,
                              ) -> Data:
    if alpha is None or epsilon is None:
        # Load adjacency matrix
        edge_index = load_adj(only_largest_cc=only_largest_cc,
                              add_self_loops=add_self_loops,
                              adj_normalization=adj_normalization,
                              dataset_name=dataset_name,
                              data_root=data_root,
                              )
        edge_weight = None
    else:
        # Load ppr matrix
        edge_index, edge_weight = load_ppr(only_largest_cc=only_largest_cc,
                                           add_self_loops=add_self_loops,
                                           adj_normalization=adj_normalization,
                                           alpha=alpha,
                                           epsilon=epsilon,
                                           dataset_name=dataset_name,
                                           data_root=data_root,
                                           )

    # Load feature matrix
    x = load_features(normalization=x_normalization,
                      dataset_name=dataset_name,
                      data_root=data_root,
                      )

    # Load label vector
    y = load_labels(dataset_name=dataset_name,
                    data_root=data_root, )

    # Create dataset
    data = Data(edge_index=edge_index,
                edge_attr=edge_weight,
                x=x,
                y=y,
                )

    return data


def load_dataset(data_root: str,
                 dataset_name: str,
                 ) -> InMemoryDataset:
    if dataset_name in ['Citeseer', 'Cora', 'Pubmed']:
        dataset = Planetoid(root=data_root,
                            name=dataset_name,
                            transform=None,
                            pre_transform=None,
                            )
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=data_root,
                           name=dataset_name,
                           transform=None,
                           pre_transform=None,
                           )
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=data_root,
                         name=dataset_name,
                         transform=None,
                         pre_transform=None,
                         )
    elif dataset_name in ['CoraFull', 'Cora_ML', 'CiteSeerFull', 'DBLP', 'PubMedFull']:
        data_root = os.path.join(data_root, 'citation_full')
        if dataset_name.endswith('Full'):
            dataset_name = dataset_name.replace('Full', '')
        dataset = CitationFull(root=data_root,
                               name=dataset_name,
                               transform=None,
                               pre_transform=None,
                               )
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=data_root,
                        name=dataset_name,
                        transform=None,
                        pre_transform=None,
                        )
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=data_root,
                                   name=dataset_name,
                                   transform=None,
                                   pre_transform=None,
                                   )
    elif dataset_name in ['Actor']:
        dataset = Actor(root=os.path.join(data_root, dataset_name),
                        transform=None,
                        pre_transform=None,
                        )
    elif dataset_name in ['Brazil', 'Europe', 'USA']:
        dataset = Airlines(root=data_root,
                           name=dataset_name,
                           transform=None,
                           pre_transform=None,
                           )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return dataset


def preprocess_dataset(dataset: Dataset,
                       only_largest_cc: bool,
                       add_self_loops: bool,
                       adj_normalization: str,
                       x_normalization: str,
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Get data
    data = dataset[0]
    edge_index = data.edge_index
    x = dataset.data.x.numpy().astype(numpy.float32)
    y = dataset.data.y

    # Restrict graph to largest connected component
    if only_largest_cc:
        graph = to_networkx(data,
                            to_undirected=False,
                            )
        largest_cc = max(networkx.weakly_connected_components(graph), key=len)
        nodes_cc = numpy.sort(list(largest_cc))
        graph = graph.subgraph(largest_cc)
        data = from_networkx(graph)

        edge_index = data.edge_index
        x = x[nodes_cc, :]
        y = y[nodes_cc]

    # Make graph undirected
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    # Add self-loops
    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index,
                                                           num_nodes=data.num_nodes,
                                                           )
    else:
        edge_weight = None

    # Normalize adjacency matrix
    edge_index, edge_weight = normalize_adj(edge_index=edge_index,
                                            edge_weight=edge_weight,
                                            num_nodes=data.num_nodes,
                                            dtype=dataset.data.x.dtype,
                                            normalization=adj_normalization,
                                            )

    # Normalize features
    x = normalize(x,
                  norm=x_normalization,
                  axis=1,
                  )
    x = torch.FloatTensor(x)

    return edge_index, edge_weight, x, y


def normalize_adj(edge_index: Tensor,
                  edge_weight: Tensor = None,
                  num_nodes: int = None,
                  dtype: torch.dtype = None,
                  normalization: str = 'sym',
                  ) -> Tuple[Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.shape[1],),
                                 dtype=dtype,
                                 device=edge_index.device,
                                 )

    if normalization is None:
        return edge_index, edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg = deg.pow_(-0.5)
        deg.masked_fill_(deg == float('inf'), 0)
        edge_weight = deg[row] * edge_weight * deg[col]
    elif normalization == 'rw':
        deg = deg.pow_(-1.)
        deg.masked_fill_(deg == float('inf'), 0)
        edge_weight = edge_weight * deg[col]
    else:
        raise ValueError(f'Unknown normalization: {normalization}')

    return edge_index, edge_weight


def get_ppr_matrix_dense(adj: numpy.ndarray,
                         alpha: float,
                         epsilon: float,
                         ) -> numpy.ndarray:
    # Compute PPR matrix
    n = adj.shape[0]
    ppr = alpha * numpy.linalg.inv(numpy.eye(n) - (1. - alpha) * adj)

    # Sparsify
    ppr[ppr < epsilon] = 0.

    # L1-normalize rows
    ppr = normalize(ppr,
                    norm='l1',
                    axis=1,
                    )

    return ppr


def add_sparse(edge_index: List[Tensor],
               edge_weight: List[Tensor],
               num_nodes: int = None,
               ) -> Tuple[Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index[0], num_nodes)

    sparse_adj = [torch.sparse.FloatTensor(edge_index_i, edge_weight_i, (num_nodes, num_nodes)) for
                  edge_index_i, edge_weight_i in zip(edge_index, edge_weight)]
    adj_sum = torch.sparse.sum(torch.stack(sparse_adj, dim=0), dim=0)

    edge_index, edge_weight = adj_sum._indices(), adj_sum._values()

    non_zero_mask = edge_weight != 0.
    edge_index = edge_index[:, non_zero_mask]
    edge_weight = edge_weight[non_zero_mask]

    return edge_index, edge_weight


def index_to_mask(index: Tensor,
                  size: int,
                  ) -> Tensor:
    mask = torch.zeros(size,
                       dtype=torch.bool,
                       device=index.device,
                       )
    mask[index] = 1

    return mask


def save_propagated_features(x: Tensor,
                             dataset_name: str,
                             data_root: str,
                             ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='propagated'
                                     )
    file_name = f'{dataset_name}.propagated'

    torch.save(x,
               os.path.join(save_dir, file_name)
               )


def load_propagated_features(dataset_name: str, data_root: str) -> Tensor:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              sub_folder='propagated'
                                              )
    file_name = f'{dataset_name}.propagated'
    return torch.load(os.path.join(preprocessed_path, file_name))


def save_prop_features_tsne(x: Tensor,
                            dataset_name: str,
                            data_root: str,
                            ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='propagated'
                                     )
    file_name = f'{dataset_name}_propagated.tsne'
    torch.save(x,
               os.path.join(save_dir, file_name)
               )


def load_prop_features_tsne(
    dataset_name: str,
    data_root: str,
) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='propagated'
                                     )
    file_name = f'{dataset_name}_propagated.tsne'
    return torch.load(os.path.join(save_dir, file_name))


def save_importance_scores(importance: Tensor, dataset_name: str, data_root: str) -> None:
    path = get_preprocessed_path(dataset_name=dataset_name,
                                 data_root=data_root,
                                 sub_folder='importance'
                                 )

    file_name = f'{dataset_name}.importance'

    torch.save(
        importance,
        os.path.join(path, file_name))


def load_importance_scores(dataset_name: str, data_root: str) -> Tensor:
    path = get_preprocessed_path(dataset_name=dataset_name,
                                 data_root=data_root,
                                 sub_folder='importance'
                                 )

    file_name = f'{dataset_name}.importance'

    return torch.load(os.path.join(path, file_name))


def load_importance_sorted(dataset_name: str, data_root: str) -> Tuple[Tensor]:
    importance = load_importance_scores(dataset_name=dataset_name, data_root=data_root)
    sorted_importance = importance.sort()
    values = sorted_importance[0].flip(0)
    indices = sorted_importance[1].flip(0)

    return values, indices


def train_DGI_and_save(path, dataset, raw_features, adj):
    batch_size = 1
    dgi_lr = 0.001
    dgi_weight_decay = 0.0
    dgi_epoch = 1000
    best_loss = 1e9
    best_iter = 0
    cnt_wait = 0
    patience = 20
    b_xent = torch.nn.BCEWithLogitsLoss()
    ft_size = raw_features.size(1)
    nb_nodes = raw_features.size(0)
    features = raw_features[numpy.newaxis]
    print("----------all Parameters-----------")
    if dataset in ['Pubmed', 'Computers', 'Photo']:
        hidden = 256
    elif dataset in ['Physics', 'CS']:
        hidden = 128
    else:
        hidden = 512
    DGI_model = DGI(ft_size, hidden, 'prelu')
    for name, param in DGI_model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    opt = optim.Adam(DGI_model.parameters(), lr=dgi_lr,
                     weight_decay=dgi_weight_decay)
    DGI_model.train()
    print('Training unsupervised model.....')
    for i in range(dgi_epoch):
        opt.zero_grad()

        perm_idx = numpy.random.permutation(nb_nodes)
        shuf_fts = features[:, perm_idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            DGI_model.cuda()
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
            features = features.cuda()
            adj = adj.cuda()

        logits = DGI_model(features, shuf_fts, adj, True, None, None, None)

        loss = b_xent(logits, lbl)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iter = i
            cnt_wait = 0
            torch.save(DGI_model.state_dict(), os.path.join(path,'best_dgi_inc11.pkl'))
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early Stopping')
            break

        loss.backward()
        opt.step()
    DGI_model.load_state_dict(torch.load(os.path.join(path,'best_dgi_inc11.pkl')))
    features, _ = DGI_model.embed(features, adj, True, None)
    features = torch.squeeze(features, 0).cpu().numpy()
    numpy.save(os.path.join(path,'dgi.npy'), features)


def load_dgi_features(path, dataset, datawrapper):
    if not os.path.exists(os.path.join(path,"dgi.npy")):
        edge_index, edge_weight = add_remaining_self_loops(edge_index=datawrapper.data.edge_index,
                                                           num_nodes=datawrapper.data.num_nodes,
                                                           )

        # Normalize adjacency matrix
        edge_index, edge_weight = normalize_adj(edge_index=edge_index,
                                                edge_weight=edge_weight,
                                                num_nodes=datawrapper.data.num_nodes,
                                                dtype=datawrapper.data.x.dtype,
                                                normalization='sym',
                                                )

        S = torch.sparse_coo_tensor(edge_index, edge_weight)

        train_DGI_and_save(
            path=path,
            dataset=dataset,
            raw_features=datawrapper.data.x,
            adj=S
        )
    return numpy.load(os.path.join(path,"dgi.npy"))


class DataWrapper:
    def __init__(self, data: Data, diffused: Data, diffused_features: Data):
        self.data = data
        self.diffused = diffused
        self.diffused_features = diffused_features

    def get_data(self):
        return self.data

    def get_diffused(self):
        return self.diffused

    def get_diffused_features(self):
        return self.diffused_features