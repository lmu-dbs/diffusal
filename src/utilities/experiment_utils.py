import os

import numpy as np
import torch
from torch import FloatTensor
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from models.pushnet import PushNetConv
from utilities.data_utils import load_labels, load_features, load_propagated_features, load_ppr_aggregated, \
    load_importance_scores, load_adj, load_dataset, preprocess_dataset, get_ppr_matrix_dense, add_sparse


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def get_dataset(
    dataset_name, 
    data_root='./data', 
    diffusion=False, 
    diffused_features=False,
    only_largest_cc = True,
    add_self_loops = True,
    x_normalization = 'l1',
    adj_normalization = 'sym',
    alphas = [5.00e-2, 1.00e-1, 2.00e-1],
    epsilon = 1.00e-5,
):
    print("*************************************************")
    print(data_root)
    if not os.path.exists(os.path.join(data_root, 'experiments')):
        print("###########")
        os.makedirs(os.path.join(data_root, 'experiments'))

    epsilon = 1.00e-5
    if dataset_name == 'CS':
        epsilon = 1.00e-4
    elif dataset_name == 'Physics':
        epsilon = 1.00e-3

    if not diffusion:
        file_name = os.path.join(data_root, 'experiments', f'{dataset_name}.pygdata')
        if os.path.exists(file_name):
            return torch.load(file_name)
        else:
            print(f'Could not find file: {file_name}')
            print('Generating instead... (this may take a WHILE for larger datasets!')

            dataset = load_dataset(dataset_name=dataset_name, data_root=data_root)
            edge_index, edge_weight, x, y = preprocess_dataset(dataset,
                                                               only_largest_cc=only_largest_cc,
                                                               add_self_loops=add_self_loops,
                                                               adj_normalization=adj_normalization,
                                                               x_normalization=x_normalization)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)
            torch.save(data, file_name)
            return data

    else:
        file_name_prefixes = [data_root, 'experiments']

        y_filename = os.path.join(*file_name_prefixes, f'{dataset_name}_y.pt')
        ppr_idx_filename = os.path.join(*file_name_prefixes, f'{dataset_name}_ppr_idx.pt')
        ppr_attr_filename = os.path.join(*file_name_prefixes, f'{dataset_name}_ppr_attr.pt')
        if diffused_features:
            prop_features_filename = os.path.join(*file_name_prefixes, f'{dataset_name}_propagated.pt')
        else:
            prop_features_filename = os.path.join(*file_name_prefixes, f'{dataset_name}_original.pt')

        if os.path.exists(y_filename) and os.path.exists(ppr_idx_filename) and \
            os.path.exists(ppr_attr_filename) and os.path.exists(prop_features_filename):

            y = torch.load(y_filename)
            edge_index = torch.load(ppr_idx_filename)
            edge_weight = torch.load(ppr_attr_filename)
            x = torch.load(prop_features_filename)

        else:
            print(f'At least one file missing for {dataset_name}.')
            print('Generating instead... (this may take a WHILE for larger datasets!')

            dataset = load_dataset(dataset_name=dataset_name, data_root=data_root)
            edge_index, edge_weight, x, y = preprocess_dataset(dataset,
                                                               only_largest_cc=only_largest_cc,
                                                               add_self_loops=add_self_loops,
                                                               adj_normalization=adj_normalization,
                                                               x_normalization=x_normalization)

            torch.save(y, y_filename)

            # Load or calculate PPR

            if os.path.exists(ppr_idx_filename) and os.path.exists(ppr_attr_filename):
                edge_index = torch.load(ppr_idx_filename)
                edge_weight = torch.load(ppr_attr_filename)
            else:
                edge_index_ppr = []
                edge_weight_ppr = []
                adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0].numpy()
                for alpha in alphas:
                    ppr_alpha = get_ppr_matrix_dense(adj=adj, alpha=alpha, epsilon=epsilon)
                    edge_index_alpha, edge_weight_alpha = dense_to_sparse(FloatTensor(ppr_alpha))
                    edge_index_ppr += [edge_index_alpha]
                    edge_weight_ppr += [edge_weight_alpha]
                edge_index, edge_weight = add_sparse(edge_index=edge_index_ppr, edge_weight=edge_weight_ppr)

                torch.save(edge_index, ppr_idx_filename)
                torch.save(edge_weight, ppr_attr_filename)

            # Load or calculate propagated feature matrix

            if os.path.exists(prop_features_filename):
                x = torch.load(prop_features_filename).cpu().float()
            else:
                if diffused_features:
                    # propagate features
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    data = Data(edge_index=edge_index, edge_attr=edge_weight, x=x, y=y).to(device)
                    propagator = PushNetConv(cached=True, batch_size_messages=16384).to(device)
                    x = propagator(data.x, data.edge_index, data.edge_attr).cpu().float()
                    torch.save(x, prop_features_filename)
                else:
                    # Simply save and return original features
                    torch.save(x, prop_features_filename)

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight).to(torch.device('cpu'))
