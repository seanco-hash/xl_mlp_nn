import multiprocessing
from os import listdir

import os
import random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import sys
import data_proccess
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch_geometric.transforms import BaseTransform
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import cross_link
import general_utils
import pdb_files_manager
from cross_link import CrossLink

ROOT_DATA_DIR = "/cs/labs/dina/seanco/xl_mlp_nn/data/"
CONNECTED_GRAPH_4F_DATASET = "graph_dataset"
NOT_CONNECTED_18F_DATASET = "onehot_graph_dataset"
SEPARATE_18F_DATASET = "separate_graphs_18f_1t_dataset"
SEPARATE_39F_ANGLES_DATASET = "separate_graphs_39f_12t_dataset"
AA_DICT = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 13: 11, 15: 12, 16: 13,
           17: 14, 18: 15, 19: 16, 21: 17, 22: 18, 23: 19, 24: 20}
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
INTRA_XL = torch.from_numpy(np.asarray([0, 1])).long
INTER_XL = torch.from_numpy(np.asarray([1, 0])).long


class FlipTransform(BaseTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data.x_a, data.x_b = data.x_b, data.x_a
            data.edge_attr_a, data.edge_attr_b = data.edge_attr_b, data.edge_attr_a
            data.edge_index_a, data.edge_index_b = data.edge_index_b, data.edge_index_a
            tmp = data.y_theta[:2].clone()
            data.y_theta[:2] = data.y_theta[2:4]
            data.y_theta[2:4] = tmp
            tmp = data.y_phi[:2].clone()
            data.y_phi[:2] = data.y_phi[2:4]
            data.y_phi[2:4] = tmp
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class TwoGraphsData(Data):
    def __init__(self, x_a=None, edge_index_a=None, edge_attr_a=None, x_b=None, edge_index_b=None,
                 edge_attr_b=None, y_ca=None, y_cb=None, y_omega=None, y_theta=None, y_phi=None,
                 linker_size=0, xl_type=INTRA_XL):
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
            return None, None, None, None, None
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
    return [[i + start_idx, j + start_idx] for i in range(n) for j in range(n) if i != j and distances[i * n + j] <= edge_th]


def crate_edge_index(features, start_idx=0, distances=None, edge_th=3):
    n = len(features)
    if distances is None:
        edges = create_complete_graph_edges(start_idx, n)
    else:
        edges = create_edges_by_dist_threshold(start_idx, distances, n, edge_th)
    if edges is None:
        return None
    return torch.as_tensor(edges, dtype=torch.long)


def res_type_to_onehot(features):
    if features.size()[1] < 19:
        return None
    input_ = features[:, -1].to(torch.long)
    try:
        input_.apply_(AA_DICT.get)
        one = F.one_hot(input_, num_classes=21)
        features = torch.cat((features[:, :-1], one), dim=1)
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
    if obj.uniport_a == obj.uniport_b:
        feat = torch.zeros(1, dtype=torch.long)
    else:
        feat = torch.ones(1, dtype=torch.long)
    feat = F.one_hot(feat, num_classes=2)
    return feat


def generate_single_xl_data(obj, feat_dict, cif_files, problems, edge_dist_th=3,  two_graphs_data=True):
    file_name = pdb_files_manager.find_pdb_file_from_xl_obj(obj, cif_files)
    uniport = file_name.split('.')[0]
    uniport = uniport.split('/')[-1]
    if uniport not in feat_dict:
        problems[0] += 1
        return None
    uniport_feat_dict = feat_dict[uniport]
    if obj.res_num_a not in uniport_feat_dict or obj.res_num_b not in uniport_feat_dict:
        problems[1] += 1
        if obj.uniport_a == obj.uniport_b:
            print(f"Problem type 1 with: {obj.uniport_a}, residues: {obj.res_num_a}, {obj.res_num_b}")
        return None
    res_a_feat = torch.from_numpy(np.stack(uniport_feat_dict[obj.res_num_a][:-1]))  # -1 - last is distance
    res_a_feat = mol2_type_to_onehot(res_a_feat)
    res_a_feat = res_type_to_onehot(res_a_feat)
    res_b_feat = torch.from_numpy(np.stack(uniport_feat_dict[obj.res_num_b][:-1]))
    res_b_feat = mol2_type_to_onehot(res_b_feat)
    res_b_feat = res_type_to_onehot(res_b_feat)
    if res_a_feat is None or res_b_feat is None:
        if obj.uniport_a == obj.uniport_b:
            print(f"Problem type 2 with: {obj.uniport_a}, residues: {obj.res_num_a}, {obj.res_num_b}")
        problems[2] += 1
        return None
    dist_a, dist_b = uniport_feat_dict[obj.res_num_a][-1], uniport_feat_dict[obj.res_num_b][-1]
    edges_a = crate_edge_index(res_a_feat, 0, dist_a, edge_dist_th)
    edge_attr_a = create_edge_atrr(edges_a, dist_a)
    if two_graphs_data:
        edges_b = crate_edge_index(res_b_feat, 0, dist_b, edge_dist_th)
        edge_attr_b = create_edge_atrr(edges_b, dist_b)
        if edges_a is None or edge_attr_a is None or edges_b is None or edge_attr_b is None:
            problems[3] += 1
            if obj.uniport_a == obj.uniport_b:
                print(f"Problem type 3 with: {obj.uniport_a}, residues: {obj.res_num_a}, {obj.res_num_b}")
            return None
        edges_a, edge_attr_a = add_self_loops(edges_a.t().contiguous(), edge_attr_a, 0)
        edges_b, edge_attr_b = add_self_loops(edges_b.t().contiguous(), edge_attr_b, 0)
        ca, cb, omega, theta, phi = create_targets(obj)
        if omega is None:
            problems[4] += 1
            return None
        data = TwoGraphsData(res_a_feat, edges_a, edge_attr_a, res_b_feat, edges_b, edge_attr_b,
                             ca, cb, omega, theta, phi, cross_link.SPACER_DICT[obj.linker_type],
                             add_xl_type_feature(obj))
    else:
        edges_b = crate_edge_index(res_b_feat, len(res_a_feat), dist_b, edge_dist_th)
        edge_attr_b = create_edge_atrr(edges_b, dist_b, len(res_a_feat))
        feat, edges, e_attr = unit_two_graphs(res_a_feat, res_b_feat, edges_a, edges_b, edge_attr_a,
                                              edge_attr_b, linker_size=None)
        edges, e_attr = add_self_loops(edges.t().contiguous(), e_attr, 0)
        data = Data(x=feat, edge_index=edges, edge_attr=e_attr, y=create_targets(obj, 'ca_only'),
                    linker=cross_link.SPACER_DICT[obj.linker_type])
    # return 1
    print(data)
    return data


def generate_graph_data(processed_xl_objects=None, feature_dict=None, pid=None, cif_files=None):
    if feature_dict is None:
        feature_dict = pdb_files_manager.get_xl_neighbors_dict()
        print(f"dict size: {len(feature_dict)}\n")
    if processed_xl_objects is None:
        processed_xl_objects = CrossLink.load_all_xl_objects(cross_link.PROCESSED_OBJ_DIR,
                                                             cross_link.PROCESSED_OBJ_DIR_PREFIX)
        processed_xl_objects = CrossLink.filter_xl_obj(processed_xl_objects, 45, 20, "any but none")
    if cif_files is None:
        cif_files = general_utils.load_obj('cif_files')
    data = []
    problems = [0, 0, 0, 0, 0]
    missing_samples = 0
    for i, xl_obj in enumerate(processed_xl_objects):
        d = generate_single_xl_data(xl_obj, feature_dict, cif_files, problems)
        if d is not None:
            data.append(d)
        else:
            missing_samples += 1
    print(f"num of data: {len(data)}, missing samples: {missing_samples}\n")
    print(problems)
    if pid is None:
        general_utils.save_obj(data, SEPARATE_39F_ANGLES_DATASET, dir_=ROOT_DATA_DIR)
    else:
        general_utils.save_obj(data, SEPARATE_39F_ANGLES_DATASET + f"_{pid}",
                               dir_=ROOT_DATA_DIR + 'parallel_output/')
    print("data saved")


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
    processed_xl_objects = CrossLink.load_all_xl_objects(cross_link.PROCESSED_OBJ_DIR,
                                                         cross_link.PROCESSED_OBJ_DIR_PREFIX)
    processed_xl_objects = CrossLink.filter_xl_obj(processed_xl_objects, 45, 20, "any but none")
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
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.get_labels(cfg, th)
        self.min_max_scaling_normalization()
        self._num_classes = cfg['num_classes']

    @property
    def raw_file_names(self):
        return [ROOT_DATA_DIR + self.dataset_name + ".pkl"]

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

    @property
    def num_classes(self):
        return self._num_classes

    def get_labels(self, cfg, th):
        self.data.y = torch.from_numpy(data_proccess.FeatDataset.get_labels_from_dist(np.asarray(self.data.y),
                                                                        cfg['num_classes'], th))

    def min_max_scaling_normalization(self):
        max_c = torch.max(self.data.x, dim=0).values
        min_c = torch.min(self.data.x, dim=0).values
        for i in range(len(self.data.x[0])):
            self.data.x[:, i] = (self.data.x[:, i] - min_c[i]) / (max_c[i] - min_c[i])

        self.data.edge_attr = self.data.edge_attr + 1
        self.data.edge_attr = 1 / self.data.edge_attr
        max_linker = torch.max(self.data.linker)
        self.data.linker = self.data.linker / max_linker
        # max_e = torch.max(self.data.edge_attr, dim=0).values
        # min_e = torch.min(self.data.edge_attr, dim=0).values
        # self.data.edge_attr[:, 0] = (self.data.edge_attr[:, 0] - min_e[0]) / (max_e[0] - min_e[0])

    def download(self):
        print("No download available\n")

    def process(self):
        # Read data into huge `Data` list.
        data_list = general_utils.load_obj(self.dataset_name, ROOT_DATA_DIR)
        data_list = self.clear_duplicates(data_list)
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

    def get_label_count(self):
        weights = np.zeros(self.num_classes)
        for i in range(len(self.data.y)):
            label = self.data.y[i]
            weights[int(label)] += 1
        return weights

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value


class XlPairGraphDataset(XlGraphDataset):
    def __init__(self, cfg, th, root=ROOT_DATA_DIR, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(cfg, th, root, transform, pre_transform, pre_filter)
        self.data.linker_size = torch.reshape(self.data.linker_size, (self.data.linker_size.size()[0], 1))

    @property
    def raw_file_names(self):
        return [ROOT_DATA_DIR + self.dataset_name + ".pkl"]

    @property
    def processed_file_names(self):
        return [self.dataset_name + '.pt']

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
        self.data.y_ca = torch.from_numpy(
            data_proccess.FeatDataset.get_labels_from_dist(np.asarray(self.data.y_ca),
                                                                cfg['num_classes'], th))
        self.data.y_cb = torch.from_numpy(
            data_proccess.FeatDataset.get_labels_from_dist(np.asarray(self.data.y_cb),
                                                                cfg['num_classes'], th))
        if cfg['num_classes'] <= 2:
            self.data.y_ca = torch.reshape(self.data.y_ca, (self.data.y_ca.shape[0], 1)).float()
            self.data.y_cb = torch.reshape(self.data.y_cb, (self.data.y_cb.shape[0], 1)).float()
        else:
            self.data.y_ca = self.data.y_ca.long()
            self.data.y_cb = self.data.y_cb.long()

        if cfg['angles'] == 'bins':
            self.get_angles_labels()

    def min_max_scaling_normalization(self):
        x = torch.cat((self.data.x_a, self.data.x_b), dim=0)
        max_c = torch.max(x, dim=0).values
        min_c = torch.min(x, dim=0).values
        for i in range(len(x[0])):
            if max_c[i] == 0:
                max_c[i] = 1
            x[:, i] = (x[:, i] - min_c[i]) / (max_c[i] - min_c[i])

        self.data.x_a, self.data.x_b = x[:len(self.data.x_a)], x[len(self.data.x_a):]

        self.data.edge_attr_a = self.data.edge_attr_a + 1
        self.data.edge_attr_a = 1 / self.data.edge_attr_a
        self.data.edge_attr_b = self.data.edge_attr_b + 1
        self.data.edge_attr_b = 1 / self.data.edge_attr_b

        max_linker = torch.max(self.data.linker_size)
        self.data.linker_size = self.data.linker_size / max_linker

    def compare_two_datasets(self, data_a, data_b):
        return torch.eq(data_a.x_a, data_b.x_a).all() and torch.eq(data_a.x_b, data_b.x_b).all() and \
               torch.eq(data_a.edge_attr_a, data_b.edge_attr_a).all() and \
               torch.eq(data_a.edge_attr_b, data_b.edge_attr_b).all() and data_a.linker_size == data_b.linker_size

    def get_comparison_key(self, data):
        return (data.x_a.size()[0], data.x_b.size()[0], data.edge_attr_a.size()[0],
                data.edge_attr_b.size()[0])

    def get_label_count(self):
        weights_a = np.zeros(self.num_classes)
        weights_b = np.zeros(self.num_classes)
        for i in range(len(self.data.y_ca)):
            label_a, label_b = self.data.y_ca[i], self.data.y_cb[i]
            weights_a[int(label_a)] += 1
            weights_b[int(label_b)] += 1
        return weights_a, weights_b

#
# data_list = general_utils.load_obj(SEPARATE_39F_ANGLES_DATASET, ROOT_DATA_DIR)
# x = 7
# data_list = XlGraphDataset.clear_duplicates(data_list)
# generate_graph_data()
# parallel_generate_graph_data()
# unit_data_pickle_files()
