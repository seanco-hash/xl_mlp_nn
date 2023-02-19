import multiprocessing
from os import listdir

import os
import random
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import sys
import data_proccess
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch.utils.data.sampler import Sampler
from typing import Callable
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import cross_link
import general_utils
import pdb_files_manager
from cross_link import CrossLink
import networkx as nx
import matplotlib.pyplot as plt
from identity import compute_identity
import copy

ROOT_DATA_DIR = "/cs/labs/dina/seanco/xl_mlp_nn/data/"
CONNECTED_GRAPH_4F_DATASET = "graph_dataset"
NOT_CONNECTED_18F_DATASET = "onehot_graph_dataset"
SEPARATE_18F_DATASET = "separate_graphs_18f_1t_dataset"
SEPARATE_39F_ANGLES_DATASET = "separate_graphs_39f_12t_dataset"
AA_DICT = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 13: 11, 15: 12, 16: 13,
           17: 14, 18: 15, 19: 16, 21: 17, 22: 18, 23: 19, 24: 20, 14: 21, 20: 22}
SS_DICT = {65: 1, 72: 0, 76: 2, 83: 1, 1: 3}
Y_DISTANCES = 0
Y_OMEGA = 1
OMEGA_SIZE = 2
OMEGA_BINS = 24
Y_THETA = 2
THETA_SIZE = 4
THETA_BINS = 24
Y_PHI = 3
PHI_SIZE = 4
PHI_BINS = 12
CA_DIST = 0
CB_DIST = 1
COS = 0
SIN = 1
INTRA_XL = torch.from_numpy(np.asarray([1, 0])).long
INTER_XL = torch.from_numpy(np.asarray([0, 1])).long
# Feature indices:
MOL2_TYPE_START = 0
MOL2_TYPE_END = 15
CHARGE_IDX = 15
ASA_IDX = 16
RADIUS_IDX = 17
RESIDUE_TYPE_START = 18
RESIDUE_TYPE_END = 41
SS_IDX_START = 41
SS_IDX_END = 45
ANCHOR_DIST_IDX = 45
ANCHOR_CB_DIST_IDX = 46
ID_ENCODING_START = 47
ID_ENCODING_END = 51
MAX_FEATURES = torch.tensor([ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
                             1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.1100,
                            44.2074,  2.2650,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
                             1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
                             1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  0.0000,
                             1.0000,  1.0000,  1.0000,  1.0000,  1.0000, 15.0000, 17.5540,  1.0000,
                             1.0000,  1.0000,  1.0000], dtype=torch.float64)
MIN_FEATURES = torch.tensor([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.7200,
                             0.0000,  1.4300,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                             0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0500,
                             0.0722,  0.0411,  0.0315], dtype=torch.float64)
MAX_LINKER_FEATURE = torch.tensor(34.4000)

# MOL2_TYPE {C2 = 0, C3 =1 , Car = 2, Ccat = 3, N2 = 4, N4 = 5, Nam = 6,
#            Nar = 7, Npl3 = 8, O2 = 9, O3 = 10, Oco2 = 11, P3 = 12, S3 = 13, UNK_MOL2_TYPE = 14 };

class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        elif isinstance(dataset, list):
            return [dataset[i].y.item() for i in range(len(dataset))]  #here the modification
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class TwoGraphsData(Data):
    def __init__(self, x_a=None, edge_index_a=None, edge_attr_a=None, x_b=None, edge_index_b=None,
                 edge_attr_b=None, y_ca=None, y_cb=None, y_omega=None, y_theta=None, y_phi=None,
                 linker_size=0, xl_type=INTRA_XL, ca_error=1):
        super().__init__()
        self.edge_index_a = edge_index_a
        self.x_a = x_a
        self.edge_attr_a = edge_attr_a
        self.edge_index_b = edge_index_b
        self.x_b = x_b
        self.edge_attr_b = edge_attr_b
        self.y_ca = y_ca
        self.y_cb = y_cb
        if y_omega is not None:
            self.y_omega = torch.from_numpy(y_omega)
        if y_theta is not None:
            self.y_theta = torch.from_numpy(y_theta)
        if y_phi is not None:
            self.y_phi = torch.from_numpy(y_phi)
        self.linker_size = linker_size
        self.xl_type = xl_type
        self.ca_error = ca_error

    @property
    def num_node_features(self) -> int:
        return self.x_a.size()[1]

    @property
    def num_edge_features(self) -> int:
        return self.edge_attr_a.size()[1]

    @property
    def num_nodes(self):
        return self.x_a.size()[0] + self.x_b.size()[0]

    @property
    def num_edges(self):
        return self.edge_attr_a.size()[0] + self.edge_attr_b.size()[0]

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def get_pandas_df_from_objects(xl_objects, feat_dict, inter_pdbs, cif_files):
    problems = [0, 0, 0, 0, 0]
    features = []
    cols = ['MOL2_TYPE', 'CHARGE', 'ASA', 'RADIUS', 'RES_TYPE', 'XL_TYPE', 'LINKER_TYPE', 'DISTANCE']
    feat_cols = ['MOL2_TYPE', 'CHARGE', 'ASA', 'RADIUS', 'RES_TYPE', 'XL_TYPE', 'DISTANCE']
    for obj in xl_objects:
        try:
            uni_a, uni_b = pdb_files_manager.find_feature_dict_keys_from_xl_obj(obj, feat_dict, problems, cif_files,
                                                                                inter_pdbs,
                                                                                feat_by_pdb=True, predict=False)
            if uni_a is None or uni_b is None or obj.res_num_a not in feat_dict[uni_a] or obj.res_num_b not in feat_dict[
                uni_b]:
                uni_a, uni_b = pdb_files_manager.find_feature_dict_keys_from_xl_obj(obj, feat_dict, problems, cif_files,
                                                                                    inter_pdbs=None, feat_by_pdb=False,
                                                                                    predict=False)
            uniport_feat_dict_a = feat_dict[uni_a]
            uniport_feat_dict_b = feat_dict[uni_b]
            res_a_feat = np.stack(uniport_feat_dict_a[obj.res_num_a][:-1])
            res_a_feat = np.sum(res_a_feat, axis=0)
            res_a_feat = np.append(res_a_feat, [obj.xl_type, cross_link.LINKER_DICT[obj.linker_type], obj.distance])
            res_b_feat = np.stack(uniport_feat_dict_b[obj.res_num_b][:-1])
            res_b_feat = np.sum(res_b_feat, axis=0)
            res_b_feat = np.append(res_b_feat, [obj.xl_type, cross_link.LINKER_DICT[obj.linker_type], obj.distance])
            features.append(res_a_feat)
            features.append(res_b_feat)
        except Exception as e:
            print(e)
            continue
    df = pd.DataFrame(data=features, columns=cols)
    df.to_pickle(f"{general_utils.OBJ_DIR}sum_features_df.pkl")
    return df, cols, feat_cols


def visualize_dataset(xl_objects, feat_dict, inter_pdbs, cif_Files, df_from_pickle=True):
    cols = ['MOL2_TYPE', 'CHARGE', 'ASA', 'RADIUS', 'RES_TYPE', 'XL_TYPE', 'LINKER_TYPE', 'DISTANCE']
    feat_cols = ['MOL2_TYPE', 'CHARGE', 'ASA', 'RADIUS', 'RES_TYPE', 'XL_TYPE', 'DISTANCE']
    if df_from_pickle:
        df = pd.read_pickle(f"{general_utils.OBJ_DIR}mean_features_df.pkl")
    else:
        df, cols, feat_cols = get_pandas_df_from_objects(xl_objects, feat_dict, inter_pdbs, cif_Files)
    features = df[feat_cols]
    # features['DISTANCE'] //= 10
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    print('t-SNE done!')

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="LINKER_TYPE",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.title("T-SNE of mean features colored by linker type")
    plt.show()
    # fig = px.scatter(
    #     projections, x=0, y=1,
    #     color=df.LINKER_TYPE, labels={'color': 'LINKER_TYPE'}
    # )
    # fig.show()


def analyze_res_type(data, n_classes=4):
    classes = [[] for i in range(n_classes)]
    for d in data:
        target = d.y_ca
        classes[target.item()].append(torch.argmax(d.x_a[:, RESIDUE_TYPE_START: RESIDUE_TYPE_END], dim=1))
        classes[target.item()].append(torch.argmax(d.x_b[:, RESIDUE_TYPE_START: RESIDUE_TYPE_END], dim=1))

    for c in classes:
        h = torch.cat(c, dim=0)
        general_utils.plot_histogram([h.numpy()], "")


def calc_node_degrees(G):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()


def visualize_graph(x, edge_index):
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(g, node_size=10, width=0.4)
    plt.show()
    calc_node_degrees(g)


def avg_degree_from_dist_mat_single_graph(distances, edge_th=3):
    n = int(np.sqrt(len(distances)))
    if n * n != len(distances):
        return None, None
    distances = distances.reshape((n, n))
    deg = np.sum(distances <= edge_th, axis=1)
    avg_deg = np.mean(deg)
    return avg_deg, n


def avg_degree_all_data(edge_th=3):
    processed_xl_objects = cross_link.get_upd_and_filter_processed_objects_pipeline()
    feat_dict = pdb_files_manager.get_xl_neighbors_dict()
    cif_files = general_utils.load_obj('cif_files')
    avg_deg = []
    avg_nodes = []
    errors = 0
    for obj in processed_xl_objects:

        # uni_a, uni_b = obj.uniport_a.split('-')[0].split('.')[0], obj.uniport_b.split('-')[0].split('.')[0]
        # if uni_a not in feat_dict or uni_b not in feat_dict:
        file_name = obj.pdb_path
        uni_a = uni_b = file_name.split('/')[-1].split('.')[0]
        if obj.chain_a != obj.chain_b or uni_a not in feat_dict:
            print(f"Problem type 0 with: {uni_a}, {uni_b}")
            return None
        uniport_feat_dict_a = feat_dict[uni_a]
        uniport_feat_dict_b = feat_dict[uni_b]
        dist_a, dist_b = uniport_feat_dict_a[obj.res_num_a][-1], uniport_feat_dict_b[obj.res_num_b][-1]
        for d in [dist_a, dist_b]:
            res_avg, n_nodes = avg_degree_from_dist_mat_single_graph(dist_a, edge_th)
            if res_avg is None:
                errors += 1
            else:
                avg_deg.append(res_avg)
                avg_nodes.append(n_nodes)

    print(f"errors: {errors}")
    print(f"Max graph average degree: {max(avg_deg)}")
    print(f"Min graph average degree: {min(avg_deg)}")
    print(f"average degree of all: {sum(avg_deg) / len(avg_deg)}")
    print(f"average nodes number: {sum(avg_nodes) / len(avg_nodes)}")


def test_pair_graph_dataloader(data, dataset_name=None):
    # original_data = general_utils.load_obj(dataset_name, ROOT_DATA_DIR)
    # data_list = [original_data[245], original_data[245]]
    loader = PyG_DataLoader(data, 2, shuffle=True,
                                  num_workers=1, follow_batch=['x_a', 'x_b'])
    batch = next(iter(loader))
    print(batch)
    print(batch.x_a_batch)
    print(batch.edge_index_a)
    print(batch.x_b_batch)
    print(batch.edge_index_b)


def unit_two_graphs_no_label_node(feat_a, feat_b, edges_a, edges_b, e_attr_a, e_attr_b):
    feat = torch.cat((feat_a, feat_b))
    edges = torch.cat((edges_a, edges_b))
    e_attr = torch.cat((e_attr_a, e_attr_b))
    return feat, edges, e_attr


def unit_two_graphs_with_label_node(feat_a, feat_b, edges_a, edges_b, e_attr_a, e_attr_b, linker_size):
    center_a, center_b = 0, len(feat_a)
    label_vertex = torch.ones((1, len(feat_a[0])))
    feat = torch.cat((feat_a, feat_b, label_vertex))
    label_edge_a = torch.LongTensor([[center_a, len(feat) - 1]])
    label_edge_b = torch.LongTensor([[center_b, len(feat) - 1]])
    edges = torch.cat((edges_a, edges_b, label_edge_a, label_edge_b))
    link_edges = torch.zeros((2, 1))
    link_edges[0, 0] = linker_size
    link_edges[1, 0] = linker_size
    e_attr = torch.cat((e_attr_a, e_attr_b, link_edges))
    return feat, edges, e_attr


def unit_two_graphs(feat_a, feat_b, edges_a, edges_b, e_attr_a, e_attr_b, linker_size):
    if linker_size is None:
        return unit_two_graphs_no_label_node(feat_a, feat_b, edges_a, edges_b, e_attr_a, e_attr_b)
    return unit_two_graphs_with_label_node(feat_a, feat_b, edges_a, edges_b, e_attr_a, e_attr_b, linker_size)


def create_targets(xl_obj, labels='all'):
    if labels == 'all':
        if not isinstance(xl_obj.omega, np.ndarray) or not isinstance(xl_obj.theta, np.ndarray):
            return [xl_obj.distance, xl_obj.cb_distance, None, None, None]
        return [xl_obj.distance, xl_obj.cb_distance, xl_obj.omega, xl_obj.theta, xl_obj.phi]
    elif labels == 'no_angles':
        return [xl_obj.distance, xl_obj.cb_distance]
    return xl_obj.distance


def create_edge_atrr(edges, dist, start_idx=0, n_edge_feat=1):
    if edges is None:
        return None
    attr = torch.zeros((len(edges), n_edge_feat))
    n = int(np.sqrt(len(dist)))
    for i in range(len(edges)):
        idx = int(edges[i, 0] - start_idx) * n + int(edges[i, 1] - start_idx)
        attr[i, 0] = dist[idx]
    return attr


def create_complete_graph_edges(start_idx, n):
    return [[i + start_idx, j + start_idx] for i in range(n) for j in range(n) if i != j]


def create_edges_by_dist_threshold(start_idx, distances, n, edge_th=3):
    if len(distances) < n * n:
        return None
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and distances[i * n + j] <= edge_th:
                edges.append([i + start_idx, j + start_idx])
    return edges
    # return [[i + start_idx, j + start_idx] for i in range(n) for j in range(n) if i != j and distances[i * n + j] <= edge_th]


def crate_edge_index(features, start_idx=0, distances=None, edge_th=3):
    n = len(features)
    if distances is None:
        edges = create_complete_graph_edges(start_idx, n)
    else:
        edges = create_edges_by_dist_threshold(start_idx, distances, n, edge_th)
    if edges is None:
        return None
    return torch.as_tensor(edges, dtype=torch.long)


def add_cb_anchor_to_feat(features, dist):
    """
    Add anchor - distance from each node to the Cb node, to get more information about node positions
    """
    n = len(features)
    ca_dist = torch.from_numpy(dist[n:2*n]).resize(n, 1)
    features = torch.cat((features, ca_dist), dim=1)
    return features


def add_ca_anchor_to_feat(features, dist):
    """
    Add anchor - distance from each node to the Ca node, to get more information about node positions
    """
    n = len(features)
    ca_dist = torch.from_numpy(dist[:n]).resize(n, 1)
    features = torch.cat((features, ca_dist), dim=1)
    return features


def add_id_encoding(features, edge_index, k_hop=4):
    """
    Add id encoding (ID-GNN-Fast, by counting cycles in each k-hop from each node.)
    :param k_hop: how many hops to count cycles
    """
    ids = compute_identity(edge_index, len(features), k_hop)
    features = torch.cat((features, ids), dim=1)
    return features


def ss_to_one_hot(features, omit=True):
    if omit:
        return features[:, :-1]
    if len(features[0]) == 5:
        features = torch.cat((features, torch.ones((len(features), 1))), dim=1)
    input_ = features[:, -1].to(torch.long)
    try:
        input_.apply_(SS_DICT.get)
        one = F.one_hot(input_, num_classes=4)
        features = torch.cat((features[:, :-1], one), dim=1)
        return features
    except Exception as e:
        print(e)
        print(input_)
        print(f"invalid SS")
        return None


def res_type_to_onehot(features, omit_ss=True, idx=-1):
    if features.size()[1] < 19:
        return None
    input_ = features[:, idx].to(torch.long)
    try:
        input_.apply_(AA_DICT.get)
        one = F.one_hot(input_, num_classes=23)
        if not omit_ss:
            features = torch.cat((features[:, :idx], one, features[:, idx + 1:]), dim=1)
        else:
            features = torch.cat((features[:, :idx], one), dim=1)
        return features
    except Exception as e:
        print(e)
        print(input_)
        return None


def mol2_type_to_onehot(features):
    input_ = features[:, 0].to(torch.long)
    one = F.one_hot(input_, num_classes=15)
    features = torch.cat((one, features[:, 1:]), dim=1)
    return features


def add_xl_type_feature(obj):
    if obj.chain_a == obj.chain_b:
        feat = torch.zeros(1, dtype=torch.long)
    else:
        feat = torch.ones(1, dtype=torch.long)
    feat = F.one_hot(feat, num_classes=2)
    return feat


def generate_single_xl_data(obj, feat_dict, cif_files, problems, edge_dist_th=5,  two_graphs_data=True,
                            inter_pdbs=None, predict=False, omit_ss=True, res_type_idx=-1):
    uni_a, uni_b = pdb_files_manager.find_feature_dict_keys_from_xl_obj(obj, feat_dict, problems, cif_files, inter_pdbs,
                                                                        feat_by_pdb=True, predict=predict)
    if uni_a is None or uni_b is None or obj.res_num_a not in feat_dict[uni_a] or obj.res_num_b not in feat_dict[uni_b]:
        uni_a, uni_b = pdb_files_manager.find_feature_dict_keys_from_xl_obj(obj, feat_dict, problems, cif_files,
                                                                            inter_pdbs=None, feat_by_pdb=False, predict=predict)
        if uni_a is None or uni_b is None or obj.res_num_a not in feat_dict[uni_a] or obj.res_num_b not in feat_dict[uni_b]:
            problems[1] += 1
            print(f"Problem type 1 with:{obj.pdb_file} {obj.uniport_a}, {obj.uniport_b} residues: {obj.res_num_a}, {obj.res_num_b}")
            return None
    uniport_feat_dict_a = feat_dict[uni_a]
    uniport_feat_dict_b = feat_dict[uni_b]
    res_a_feat = torch.from_numpy(np.stack(uniport_feat_dict_a[obj.res_num_a][:-1]))  # -1 - last is distance
    res_a_feat = ss_to_one_hot(res_a_feat, omit_ss)
    res_a_feat = mol2_type_to_onehot(res_a_feat)
    res_a_feat = res_type_to_onehot(res_a_feat, omit_ss, res_type_idx)
    res_b_feat = torch.from_numpy(np.stack(uniport_feat_dict_b[obj.res_num_b][:-1]))
    res_b_feat = ss_to_one_hot(res_b_feat, omit_ss)
    res_b_feat = mol2_type_to_onehot(res_b_feat)
    res_b_feat = res_type_to_onehot(res_b_feat, omit_ss, res_type_idx)
    if res_a_feat is None or res_b_feat is None:
        print(f"Problem type 2 with:{obj.pdb_file} {obj.uniport_a}, residues: {obj.res_num_a}, {obj.res_num_b}")
        problems[2] += 1
        return None
    dist_a, dist_b = uniport_feat_dict_a[obj.res_num_a][-1], uniport_feat_dict_b[obj.res_num_b][-1]
    edges_a = crate_edge_index(res_a_feat, 0, dist_a, edge_dist_th)
    edge_attr_a = create_edge_atrr(edges_a, dist_a)
    if two_graphs_data:
        edges_b = crate_edge_index(res_b_feat, 0, dist_b, edge_dist_th)
        edge_attr_b = create_edge_atrr(edges_b, dist_b)
        if edges_a is None or edge_attr_a is None or edges_b is None or edge_attr_b is None:
            problems[3] += 1
            print(f"Problem type 3 with: {obj.uniport_a}, residues: {obj.res_num_a}, {obj.res_num_b}")
            return None
        edges_a, edge_attr_a = add_self_loops(edges_a.t().contiguous(), edge_attr_a, 0)
        edges_b, edge_attr_b = add_self_loops(edges_b.t().contiguous(), edge_attr_b, 0)
        res_a_feat = add_ca_anchor_to_feat(res_a_feat, dist_a)
        res_a_feat = add_cb_anchor_to_feat(res_a_feat, dist_a)
        res_b_feat = add_ca_anchor_to_feat(res_b_feat, dist_b)
        res_b_feat = add_cb_anchor_to_feat(res_b_feat, dist_b)
        res_a_feat = add_id_encoding(res_a_feat, edges_a)
        res_b_feat = add_id_encoding(res_b_feat, edges_b)
        ca, cb, omega, theta, phi = create_targets(obj)
        if omega is None:
            cb = ca
            print(f"Problem type 4 with: {obj.uniport_a}, residues: {obj.res_num_a}, {obj.res_num_b}")
            print(f"{ca}, {cb}, {omega}, {theta}, {phi}")
            problems[4] += 1
            omega, theta, phi = np.ones(2), np.ones(4), np.ones(4)
        # visualize_graph(res_a_feat, edges_a)
        data = TwoGraphsData(res_a_feat, edges_a, edge_attr_a, res_b_feat, edges_b, edge_attr_b,
                             ca, cb, omega, theta, phi, cross_link.SPACER_DICT[obj.linker_type],
                             add_xl_type_feature(obj), obj.error)
    else:
        edges_b = crate_edge_index(res_b_feat, len(res_a_feat), dist_b, edge_dist_th)
        edge_attr_b = create_edge_atrr(edges_b, dist_b, len(res_a_feat))
        feat, edges, e_attr = unit_two_graphs(res_a_feat, res_b_feat, edges_a, edges_b, edge_attr_a,
                                              edge_attr_b, linker_size=None)
        edges, e_attr = add_self_loops(edges.t().contiguous(), e_attr, 0)
        data = Data(x=feat, edge_index=edges, edge_attr=e_attr, y=create_targets(obj, 'ca_only'),
                    linker=cross_link.SPACER_DICT[obj.linker_type], ca_error=obj.error)
    # print(data)
    return data


def generate_graph_data(processed_xl_objects=None, feature_dict=None, pid=None, cif_files=None,
                        data_name=SEPARATE_39F_ANGLES_DATASET, pdb_path=None, edge_dist_th=3, inter_pdbs=None,
                        save=True, data=None, predict=False, omit_ss=True, res_type_idx=-1):
    if feature_dict is None:
        feature_dict = pdb_files_manager.get_xl_neighbors_dict()
        print(f"dict size: {len(feature_dict)}\n")
    if processed_xl_objects is None:
        processed_xl_objects = cross_link.get_upd_and_filter_processed_objects_pipeline()
    if cif_files is None:
        cif_files = general_utils.load_obj('cif_files')
    if data is None:
        data = []
    problems = [0, 0, 0, 0, 0]
    missing_samples = 0
    # processed_xl_objects = pdb_files_manager.filter_objects_from_list_by_pdb(processed_xl_objects, inter_pdbs)
    for i, xl_obj in enumerate(processed_xl_objects):
        try:
            d = generate_single_xl_data(xl_obj, feature_dict, cif_files, problems, edge_dist_th, True, inter_pdbs, predict, omit_ss=omit_ss, res_type_idx=res_type_idx)
            if d is not None:
                data.append(d)
            else:
                missing_samples += 1
            if i % 1000 == 0:
                print(i)
        except Exception as e:
            print(e)
    print(f"num of data: {len(data)}, missing samples: {missing_samples}\n")
    print(problems)
    if save:
        if pid is None:
            general_utils.save_obj(data, data_name, dir_=ROOT_DATA_DIR)
        else:
            general_utils.save_obj(data, data_name + f"_{pid}",
                                   dir_=ROOT_DATA_DIR + 'parallel_output/')
    print("data saved")
    return data


def unit_data_pickle_files():
    data = []
    dir_ = ROOT_DATA_DIR + 'parallel_output/'
    for file in listdir(dir_):
        cur_data = general_utils.load_obj(file.split('.')[0], dir_)
        data += cur_data
    general_utils.save_obj(data, SEPARATE_39F_ANGLES_DATASET, dir_=ROOT_DATA_DIR)
    for file in listdir(dir_):
        os.remove(dir_ + file)


def parallel_generate_graph_data():
    feature_dict = pdb_files_manager.get_xl_neighbors_dict()
    print(f"dict size: {len(feature_dict)}\n")
    processed_xl_objects = cross_link.get_upd_and_filter_processed_objects_pipeline()
    cif_files = general_utils.load_obj('cif_files')
    available_cpus = len(os.sched_getaffinity(0)) - 1
    print("available cpus: ", available_cpus, flush=True)
    start_idx = 0
    step = int(len(processed_xl_objects) / (available_cpus - 1))
    processes = []
    try:
        for pid in range(available_cpus):
            end = min(start_idx + step, len(processed_xl_objects))
            p = multiprocessing.Process(target=generate_graph_data,
                                        args=(processed_xl_objects[start_idx: end], feature_dict, pid, cif_files))
            p.start()
            processes.append(p)
            start_idx += step
        for p in processes:
            p.join()
    except Exception as e:
        print(e)


class XlGraphDataset(InMemoryDataset):
    def __init__(self, cfg, th, root=ROOT_DATA_DIR, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = cfg['dataset']
        if not os.path.isdir(f"{ROOT_DATA_DIR}{self.dataset_name}/"):
            os.makedirs(f"{ROOT_DATA_DIR}{self.dataset_name}/")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.get_labels(cfg, th)
        self.min_max_scaling_normalization(cfg)
        self._num_classes = cfg['num_classes']

    @property
    def raw_file_names(self):
        return [ROOT_DATA_DIR + self.dataset_name + ".pkl"]

    @property
    def processed_file_names(self):
        return [f"{ROOT_DATA_DIR}{self.dataset_name}/{self.dataset_name}.pt"]

    @property
    def num_classes(self):
        return self._num_classes

    def get_labels(self, cfg, th):
        self.data.y = torch.from_numpy(data_proccess.FeatDataset.get_labels_from_dist(np.asarray(self.data.y),
                                                                        cfg['num_classes'], th))

    def min_max_scaling_normalization(self, cfg=None):
        if cfg is None or cfg['data_type'] == 'Train':
            max_c = torch.max(self.data.x, dim=0).values
            min_c = torch.min(self.data.x, dim=0).values
            max_linker = torch.max(self.data.linker)
        else:
            max_c = MAX_FEATURES
            min_c = MIN_FEATURES
            max_linker = MAX_LINKER_FEATURE
        for i in range(len(self.data.x[0])):
            self.data.x[:, i] = (self.data.x[:, i] - min_c[i]) / (max_c[i] - min_c[i])

        self.data.edge_attr = self.data.edge_attr + 1
        self.data.edge_attr = 1 / self.data.edge_attr

        self.data.linker = self.data.linker / max_linker
        # max_e = torch.max(self.data.edge_attr, dim=0).values
        # min_e = torch.min(self.data.edge_attr, dim=0).values
        # self.data.edge_attr[:, 0] = (self.data.edge_attr[:, 0] - min_e[0]) / (max_e[0] - min_e[0])

    def download(self):
        print("No download available\n")

    @staticmethod
    def over_sample_by_feat(data_list, n_copies):
        add_to_list = []
        for d in data_list:
            if d.xl_type == INTER_XL:
                for i in range(n_copies):
                    add_to_list.append(copy.deepcopy(d))

    def process(self):
        # Read data into huge `Data` list.
        data_list = general_utils.load_obj(self.dataset_name, ROOT_DATA_DIR)
        # data_list = self.clear_duplicates(data_list)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def compare_two_datasets(self, data_a, data_b):
        return torch.eq(data_a.x, data_b.x).all() and torch.eq(data_a.edge_attr, data_b.edge_attr).all() and \
               data_a.linker == data_b.linker

    def get_comparison_key(self, data):
        return (data.num_nodes, data.num_edges)

    def clear_duplicates(self, data_list):
        similar_datasets = {}
        for i, data in enumerate(data_list):
            key = self.get_comparison_key(data)
            if key not in similar_datasets:
                similar_datasets[key] = []
            similar_datasets[key].append(i)

        to_del = set()
        for similar_list in similar_datasets.values():
            for k in range(len(similar_list)):
                if similar_list[k] not in to_del:
                    for l in range(k+1, len(similar_list)):
                        if self.compare_two_datasets(data_list[similar_list[k]], data_list[similar_list[l]]):
                            to_del.add(similar_list[l])

        for i in sorted(list(to_del), reverse=True):
            del data_list[i]
        return data_list

    def get_label_count(self, train_idx=None):
        if train_idx is None:
            train_idx = torch.tensor(range(len(self.data.y)))
        weights = np.zeros(self.num_classes)
        for i in range(len(self.data.y[train_idx])):
            label = self.data.y[train_idx][i]
            weights[int(label)] += 1
        return weights

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value


class XlPairGraphDataset(XlGraphDataset):
    def __init__(self, cfg, th, root=ROOT_DATA_DIR, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(cfg, th, root, transform, pre_transform, pre_filter)
        if len(self.data.linker_size.size()) > 0:
            self.data.linker_size = torch.reshape(self.data.linker_size, (self.data.linker_size.size()[0], 1))
        else:
            self.data.linker_size = torch.reshape(self.data.linker_size, (1, 1))
        self.remove_features(cfg)

    @property
    def raw_file_names(self):
        return [ROOT_DATA_DIR + self.dataset_name + ".pkl"]

    @property
    def processed_file_names(self):
        return [f"{ROOT_DATA_DIR}{self.dataset_name}/{self.dataset_name}.pt"]

    def process(self):
        super(XlPairGraphDataset, self).process()

    def get_y(self, data):
        return [data.y_ca, data.y_cb, data.y_omega, data.y_theta, data.y_phi]

    def get_angles_labels(self):
        omega = torch.reshape(self.data.y_omega, (self.data.y_omega.shape[0] // 2, 2))
        theta = torch.reshape(self.data.y_theta, (self.data.y_theta.shape[0] // 4, 4))
        phi = torch.reshape(self.data.y_phi, (self.data.y_phi.shape[0] // 4, 4))
        omega_a = torch.rad2deg_(torch.atan2(omega[:, 1], omega[:, 0])) + 180
        omega_a = torch.reshape(torch.remainder(omega_a, OMEGA_BINS).int(), (omega_a.shape[0], 1))

        phi_a = torch.rad2deg_(torch.atan2(phi[:, 1], phi[:, 0]))
        phi_a = torch.reshape(torch.remainder(phi_a, PHI_BINS).int(), (phi_a.shape[0], 1))
        phi_b = torch.rad2deg_(torch.atan2(phi[:, 3], phi[:, 2]))
        phi_b = torch.reshape(torch.remainder(phi_b, PHI_BINS).int(), (phi_b.shape[0], 1))

        theta_a = torch.rad2deg_(torch.atan2(theta[:, 1], theta[:, 0])) + 180
        theta_a = torch.reshape(torch.remainder(theta_a, THETA_BINS).int(), (theta_a.shape[0], 1))
        theta_b = torch.rad2deg_(torch.atan2(theta[:, 3], theta[:, 2])) + 180
        theta_b = torch.reshape(torch.remainder(theta_b, THETA_BINS).int(), (theta_b.shape[0], 1))

        self.data.y_phi = torch.reshape(torch.cat((phi_a, phi_a, phi_b, phi_b), dim=1), self.data.y_phi.shape)
        self.data.y_theta = torch.reshape(torch.cat((theta_a, theta_a, theta_b, theta_b), dim=1),
                                          self.data.y_theta.shape)
        self.data.y_omega = torch.reshape(torch.cat((omega_a, omega_a), dim=1),
                                          self.data.y_omega.shape)

    def get_labels(self, cfg, th):
        y_ca_labels, th = data_proccess.FeatDataset.get_labels_from_dist(np.asarray(self.data.y_ca),
                                                                         cfg['num_classes'], th)
        cfg['distance_th_classification'] = th
        self.data.y_ca = torch.from_numpy(y_ca_labels)
        y_cb_labels, th = data_proccess.FeatDataset.get_labels_from_dist(np.asarray(self.data.y_cb),
                                                                         cfg['num_classes'], th)
        self.data.y_cb = torch.from_numpy(y_cb_labels)
        if cfg['num_classes'] <= 2:
            self.data.y_ca = torch.reshape(self.data.y_ca, (self.data.y_ca.shape[0], 1)).float()
            self.data.y_cb = torch.reshape(self.data.y_cb, (self.data.y_cb.shape[0], 1)).float()
        else:
            self.data.y_ca = self.data.y_ca.long()
            self.data.y_cb = self.data.y_cb.long()

        if cfg['angles'] == 'bins':
            self.get_angles_labels()

    @staticmethod
    def get_dist_from_label(label, th):
        if label < len(th):
            max_dist = th[label]
        else:
            max_dist = 45
        if label > 0:
            min_dist = th[label - 1]
        else:
            min_dist = 0
        return min_dist, max_dist

    def remove_features(self, cfg):
        if 'features' in cfg:
            keep_features = cfg['features']
            if 'mol2' not in keep_features:
                self.data.x_a[:, MOL2_TYPE_START: MOL2_TYPE_END] = 0
                self.data.x_b[:, MOL2_TYPE_START: MOL2_TYPE_END] = 0
            if 'charge' not in keep_features:
                self.data.x_a[:, CHARGE_IDX] = 0
                self.data.x_b[:, CHARGE_IDX] = 0
            if 'asa' not in keep_features:
                self.data.x_a[:, ASA_IDX] = 0
                self.data.x_b[:, ASA_IDX] = 0
            if 'radius' not in keep_features:
                self.data.x_a[:, RADIUS_IDX] = 0
                self.data.x_b[:, RADIUS_IDX] = 0
            if 'res_type' not in keep_features:
                self.data.x_a[:, RESIDUE_TYPE_START: RESIDUE_TYPE_END] = 0
                self.data.x_b[:, RESIDUE_TYPE_START: RESIDUE_TYPE_END] = 0
            if 'anchor' not in keep_features:
                self.data.x_a[:, ANCHOR_DIST_IDX] = 0
                self.data.x_b[:, ANCHOR_DIST_IDX] = 0
            if 'anchor_cb' not in keep_features:
                self.data.x_a[:, ANCHOR_CB_DIST_IDX] = 0
                self.data.x_b[:, ANCHOR_CB_DIST_IDX] = 0
            if 'id' not in keep_features:
                self.data.x_a[:, ID_ENCODING_START: ID_ENCODING_END] = 0
                self.data.x_b[:, ID_ENCODING_START: ID_ENCODING_END] = 0
            if 'edges' not in keep_features:
                self.data.edge_attr_a[:] = 1
                self.data.edge_attr_b[:] = 1
            if 'ss' not in keep_features:
                self.data.x_a[:, SS_IDX_START: SS_IDX_END] = 0
                self.data.x_b[:, SS_IDX_START: SS_IDX_END] = 0

    def min_max_scaling_normalization(self, cfg=None):
        x = torch.cat((self.data.x_a, self.data.x_b), dim=0)
        if cfg is None or cfg['data_type'] == 'Train':
            max_c = torch.max(x, dim=0).values
            min_c = torch.min(x, dim=0).values
            max_linker = torch.max(self.data.linker_size)
        else:
            max_c = MAX_FEATURES
            min_c = MIN_FEATURES
            max_linker = MAX_LINKER_FEATURE

        for i in range(len(x[0]) - 4):
            if max_c[i] == 0:
                max_c[i] = 1
            x[:, i] = (x[:, i] - min_c[i]) / (max_c[i] - min_c[i])

        self.data.x_a, self.data.x_b = x[:len(self.data.x_a)], x[len(self.data.x_a):]

        self.data.edge_attr_a = self.data.edge_attr_a + 1
        self.data.edge_attr_a = 1 / self.data.edge_attr_a
        self.data.edge_attr_b = self.data.edge_attr_b + 1
        self.data.edge_attr_b = 1 / self.data.edge_attr_b

        self.data.linker_size = self.data.linker_size / max_linker

        # self.data.ca_error[self.data.ca_error == cross_link.INVALID_ERROR_VALUE] = cross_link.MEAN_PAE_ERROR
        # self.data.ca_error[self.data.ca_error < 1] = 1
        # self.data.ca_error = 1 / self.data.ca_error

    def compare_two_datasets(self, data_a, data_b):
        return torch.eq(data_a.x_a, data_b.x_a).all() and torch.eq(data_a.x_b, data_b.x_b).all() and \
               torch.eq(data_a.edge_attr_a, data_b.edge_attr_a).all() and \
               torch.eq(data_a.edge_attr_b, data_b.edge_attr_b).all() and data_a.linker_size == data_b.linker_size

    def get_comparison_key(self, data):
        return (data.x_a.size()[0], data.x_b.size()[0], data.edge_attr_a.size()[0],
                data.edge_attr_b.size()[0])

    def get_label_count(self,  train_idx=None):
        weights_a = np.zeros(self.num_classes)
        weights_b = np.zeros(self.num_classes)
        for i in range(len(self.data.y_ca[train_idx])):
            label_a, label_b = self.data.y_ca[train_idx][i], self.data.y_cb[train_idx][i]
            weights_a[int(label_a)] += 1
            weights_b[int(label_b)] += 1
        return weights_a, weights_b

    @staticmethod
    def get_ca_labels(dataset):
        idx = dataset.indices
        return dataset.dataset.data.y_ca[idx]
