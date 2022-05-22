import numpy as np
from torch.utils.data import Dataset
import sys
from torchvision import transforms
import random
from sklearn.metrics.pairwise import cosine_distances
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import cross_link
import matplotlib.pyplot as plt


XL_PROJ_DIR = '/cs/labs/dina/seanco/xl_parser/'
SCANNET_PROJ_DIR = '/cs/labs/dina/seanco/ScanNet/ScanNet/'
MLP_PROJ_DIRD = '/cs/labs/dina/seanco/xl_mlp_nn/'


def save_np_obj(obj, name, folder):
    with open(folder + 'obj/' + name + '.npy', 'wb') as f:
        np.save(f, obj)


def load_np_obj(name, folder):
    with open(folder + 'obj/' + name + '.npy', 'rb') as f:
        return np.load(f, allow_pickle=True)


def extract_outlier_feature_names():
    features = load_np_obj('processed_features', SCANNET_PROJ_DIR)
    names = load_np_obj('processed_keys', SCANNET_PROJ_DIR)
    names = names[(features > 20).any(axis=1)]
    features = features[(features > 20).any(axis=1)]
    x = 0


def features_values_histogram(features):
    # features = load_np_obj('processed_features_trainset', MLP_PROJ_DIRD)
    features = FeatDataset.clear_redundant_columns(features)
    features = features[:, :-1]
    x = features.flatten()
    # plt.yscale('log', nonposy='clip')
    plt.hist(x, 200, log=True)
    plt.show()


def check_similar_features():
    features = load_np_obj('processed_features_trainset', MLP_PROJ_DIRD)
    print("num features: ", len(features))
    dist = load_np_obj('processed_dist_trainset', MLP_PROJ_DIRD)
    norm = np.sqrt((features ** 2).sum(-1))
    # cosdist = 1 - np.dot(features, features.T) / np.sqrt(norm[np.newaxis, :] * norm[np.newaxis, :])
    cosdist = cosine_distances(features)
    similar_idx = np.asarray(np.where(np.logical_and(cosdist < 0.05, cosdist > 0))).T
    avg = 0.0
    avg_counter = 0
    oposite_labels = {}
    for pair in similar_idx:
        if (dist[pair[0]] < 18 <= dist[pair[1]]) or (dist[pair[0]] >= 18 > dist[pair[1]]):
            if pair[0] not in oposite_labels:
                oposite_labels[pair[0]] = set()
            oposite_labels[pair[0]].add(pair[1])
            if pair[1] not in oposite_labels:
                oposite_labels[pair[1]] = set()
                oposite_labels[pair[1]].add(pair[0])
            avg += abs(dist[pair[0]] - dist[pair[1]])
            avg_counter += 1
    print("avg distance of oposite label features: ", avg / avg_counter)
    # cosdist_flat = cosdist[cosdist != 0].flatten()
    print("num op: ", len(oposite_labels))
    # plt.hist(cosdist_flat, bins=11, range=(0, 1), log=True)
    # plt.show()


def check_duplicated_features():
    features = load_np_obj('processed_features_trainset', MLP_PROJ_DIRD)
    dist = load_np_obj('processed_dist_trainset', MLP_PROJ_DIRD)
    # labels = load_np_obj('processed_labels_trainset', MLP_PROJ_DIRD)
    print(f"initial features shape: {features.shape}")
    dup = 0
    dist_mismatch = 0
    to_del = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if (features[i] == features[j]).all():
                dup += 1
                to_del.append(j)
                if dist[i] != dist[j]:
                    dist_mismatch += 1
                    print(f"dist1: {dist[i]}, dist2: {dist[j]}")
    features = np.delete(features, to_del, axis=0)
    dist = np.delete(dist, to_del, axis=0)
    # labels = np.delete(labels, to_del, axis=0)
    print(f"final features shape: {features.shape}")
    print("equal features: ", dup)
    print("distance mismatch: ", dist_mismatch)
    save_np_obj(features, 'processed_features_trainset', MLP_PROJ_DIRD)
    save_np_obj(dist, 'processed_dist_trainset', MLP_PROJ_DIRD)
    # save_np_obj(labels, 'processed_labels_trainset', MLP_PROJ_DIRD)


def calc_mean_and_std():
    features = load_np_obj('processed_features_trainset', MLP_PROJ_DIRD)
    labels = load_np_obj('processed_dist_trainset', MLP_PROJ_DIRD)
    lysine = features[0, :32]
    all_atoms = features[:, :32]
    all_atoms = np.vstack((all_atoms, features[:, 224:256]))
    tmp = all_atoms[(all_atoms == lysine).all()]
    unique, counts = np.unique(features, return_counts=True)
    d = dict(zip(unique, counts))
    d = sorted(d.items(), key=lambda pair: pair[0])
    print(features[:, :-1].mean(), features[:, :-1].std())


def distances_to_one_hot():
    distances = load_np_obj('best_dsso_dist', SCANNET_PROJ_DIR)
    percentiles = np.percentile(distances, [1.0/3.0, 2.0/3.0])
    one_hot = np.zeros((distances.shape[0], 3), dtype=np.float32)
    for i in range(distances.shape[0]):
        dist = distances[i]
        if dist <= percentiles[0]:
            vec = [1, 0, 0]
        elif dist <= percentiles[1]:
            vec = [0, 1, 0]
        else:
            vec = [0, 0, 1]
        one_hot[i] = np.asarray(vec, dtype=np.float32)

    save_np_obj(one_hot, 'labels', MLP_PROJ_DIRD)


def distances_to_categories(cfg):
    distances = load_np_obj(cfg['RAW_DATA']['DIST_LIST'], SCANNET_PROJ_DIR)
    percentiles = np.percentile(distances, [33, 66])
    categories = np.zeros((distances.shape[0]), dtype=np.int64)
    for i in range(distances.shape[0]):
        dist = distances[i]
        if dist <= percentiles[0]:
            cat = 0
        elif dist <= percentiles[1]:
            cat = 1
        else:
            cat = 2
        categories[i] = cat

    save_np_obj(categories, cfg['RAW_DATA']['LABEL_LIST'], MLP_PROJ_DIRD)


def create_test_set(cfg):
    distances = load_np_obj(cfg['RAW_DATA']['DIST_LIST'], SCANNET_PROJ_DIR)
    # labels = load_np_obj(cfg['RAW_DATA']['LABEL_LIST'], MLP_PROJ_DIRD)
    features = load_np_obj(cfg['RAW_DATA']['FEATURES_LIST'], SCANNET_PROJ_DIR)
    test_set_size = int(features.shape[0] / 10)
    index = np.random.choice(features.shape[0], test_set_size, replace=False)
    test_features = np.copy(features[index])
    # test_labels = np.copy(labels[index])
    test_dist = np.copy(distances[index])
    train_feat = np.delete(features, index, axis=0)
    # train_labels = np.delete(labels, index)
    train_dist = np.delete(distances, index)
    save_np_obj(test_features, cfg['TEST']['FEATURES'], MLP_PROJ_DIRD)
    # save_np_obj(test_labels, cfg['TEST']['LABELS'], MLP_PROJ_DIRD)
    save_np_obj(train_feat, cfg['DATA']['FEATURES'], MLP_PROJ_DIRD)
    # save_np_obj(train_labels, cfg['DATA']['LABELS'], MLP_PROJ_DIRD)
    save_np_obj(train_dist, cfg['DATA']['DISTANCES'], MLP_PROJ_DIRD)
    save_np_obj(test_dist, cfg['TEST']['DISTANCES'], MLP_PROJ_DIRD)


class FeatDataset(Dataset):

    @staticmethod
    def get_labels_by_thresholds(dist, thresholds):
        dist[np.logical_and(0 <= dist, dist < thresholds[0])] = 0
        cat = 1
        for i in range(1, len(thresholds)):
            dist[np.logical_and(thresholds[i-1] <= dist, dist < thresholds[i])] = cat
            cat += 1
        dist[thresholds[-1] <= dist] = cat
        return dist

    @staticmethod
    def get_labels_from_dist(dist, n_classes, th=None):
        if th is None or th == 'None' or len(th) < n_classes - 1:
            fraction = 100 / n_classes
            perc_arr = [i * fraction for i in range(1, n_classes)]
            percentiles = np.percentile(dist, perc_arr)
        else:
            percentiles = th
        return FeatDataset.get_labels_by_thresholds(dist, percentiles)

    @staticmethod
    def delete_similar_features(features, labels, clear_th=0.05):
        cosdist = cosine_distances(features)
        similar_idx = np.asarray(np.where(np.logical_and(cosdist < clear_th, cosdist > 0))).T
        oposite_labels = set()
        for pair in similar_idx:
            if labels[pair[0]] != labels[pair[1]] and features[pair[0], -1] == features[pair[1], -1]:
                to_del = min(pair[0], pair[1])
                oposite_labels.add(to_del)
        features = np.delete(features, list(oposite_labels), axis=0)
        labels = np.delete(labels, list(oposite_labels), axis=0)
        return features, labels

    @staticmethod
    def clear_outliers(features, labels, th=10):
        labels = labels[(features < th).all(axis=1)]
        features = features[(features < th).all(axis=1)]
        return features, labels

    @staticmethod
    def clear_redundant_columns(features):
        n_features = len(features[0]) // 2
        stacked = np.vstack((features[:, :n_features], features[:, n_features:-1]))
        stacked = stacked[:, np.max(stacked, axis=0) != 0]
        half = int(len(stacked) / 2)
        features = np.concatenate((stacked[:half], stacked[half:], features[:,-1].reshape((len(features), 1))), axis=1)
        return features

    @staticmethod
    def min_max_scaling_normalization(features):
        stacked = np.vstack((features[:, :224], features[:, 224:448]))
        max_ = np.max(stacked, axis=0)
        min_ = np.min(stacked, axis=0)
        stacked = (stacked - min_) / (max_ - min_)
        half = int(len(stacked) / 2)
        features[:, :224] = stacked[:half]
        features[:, 224:448] = stacked[half:]
        # features = features[:, ~np.isnan(features).any(axis=0)]
        features[:, -1] = (features[:, -1] - np.min(features[:, -1])) / (np.max(features[:, -1]) - np.min(features[:, -1]))
        features[np.isnan(features)] = 0
        return features

    @staticmethod
    def max_scaling_normalization(features):
        n_features = len(features[0]) // 2
        stacked = np.vstack((features[:, :n_features], features[:, n_features:-1]))
        m = np.max(stacked, axis=0)
        stacked = stacked / m
        stacked = stacked[:, ~np.isnan(stacked).all(axis=0)]
        half = int(len(stacked) / 2)
        features = np.concatenate((stacked[:half], stacked[half:], features[:,-1].reshape((len(features), 1))), axis=1)
        # features[:, :n_features] = stacked[:half]
        # features[:, n_features:-1] = stacked[half:]
        features[:, -1] = features[:, -1] / np.max(features[:, -1])
        features[np.isnan(features)] = 0
        return features

    @staticmethod
    def tanh_estimator_normalization(features):
        stacked = np.vstack((features[:, :224], features[:, 224:448]))
        m = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        stacked = 0.5 * (np.tanh(0.01 * ((stacked - m) / std)) + 1)
        half = int(len(stacked) / 2)
        features[:, :224] = stacked[:half]
        features[:, 224:448] = stacked[half:]
        # features = features[:, ~np.isnan(features).any(axis=0)]
        features[np.isnan(features)] = 0
        spacer_col = features[:, -1]
        features[:, -1] = 0.5 * (np.tanh(0.01 * ((spacer_col - np.mean(spacer_col)) / np.std(spacer_col))) + 1)
        return features

    @staticmethod
    def filter_features(features, labels, cfg):
        if cfg['DATA']['FILTER'] == 'dsso_dss':
            labels = labels[features[:, -1] != cross_link.BDP_NHP_SPACER / 100]
            features = features[features[:, -1] != cross_link.BDP_NHP_SPACER / 100]
        if cfg['DATA']['FILTER'] == 'lysine':
            tmp1 = features[:, :32]
            lysine_feat = features[0, :32]
            labels = labels[(tmp1 == lysine_feat).all(axis=1)]
            features = features[(tmp1 == lysine_feat).all(axis=1)]
            tmp2 = features[:, 224:256]
            labels = labels[(tmp2 == lysine_feat).all(axis=1)]
            features = features[(tmp2 == lysine_feat).all(axis=1)]
            features = np.concatenate((features[:, 32:224], features[:, 256:]), axis=1)
        return features, labels

    @staticmethod
    def get_features(feat_path, labels_path, dataset_type, is_train, loss_type, dist_path, cfg):
        dist = load_np_obj(dist_path, MLP_PROJ_DIRD)
        labels = FeatDataset.get_labels_from_dist(dist, cfg['MODEL']['NUM_CLASSES'],
                                                  cfg['MODEL']['DISTANCE_TH_CLASSIFICATION'])
        features = load_np_obj(feat_path, MLP_PROJ_DIRD)
        features, labels = FeatDataset.filter_features(features, labels, cfg)
        # features = FeatDataset.tanh_estimator_normalization(features)
        # features = FeatDataset.max_scaling_normalization(features)
        features = FeatDataset.clear_redundant_columns(features)
        features, labels = FeatDataset.clear_outliers(features, labels)
        features, labels = FeatDataset.delete_similar_features(features, labels, cfg['DATA']['SIMILARITY_CLEAR'])
        # features_values_histogram(features)
        # features = FeatDataset.min_max_scaling_normalization(features)
        ten_percent = int(0.1 * features.shape[0])
        if dataset_type == 'TEST':
            return features, labels
        elif is_train:
            return features[ten_percent:], labels[ten_percent:]
        return features[:ten_percent], labels[:ten_percent]

    def __init__(self, cfg, is_train=True):
        features_path = cfg['DATA']['FEATURES']
        labels_path = cfg['DATA']['LABELS']
        distances_path = cfg['DATA']['DISTANCES']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.cfg = cfg
        self.features, self.labels = FeatDataset.get_features(features_path, labels_path, cfg['DATA']['TYPE'],
                                                              is_train, cfg['LOSS']['TYPE'],
                                                              distances_path, cfg)
        self.transform = self.get_transform()

    @staticmethod
    def flip_transform(batch):
        if random.random() < 0.5:
            half = int(len(batch) / 2)
            batch[:half], batch[half:] = batch[half:], batch[:half]

    def get_transform(self):
        if self.cfg['DATA']['TRANSFORM'] == 'normalize':
            return [transforms.Compose([transforms.Normalize(mean=self.cfg['DATA']['MEAN'],
                                                             std=self.cfg['DATA']['STD'], inplace=True)])]
        if self.cfg['DATA']['TRANSFORM'] == 'flip':
            return [FeatDataset.flip_transform]
        if self.cfg['DATA']['TRANSFORM'] == 'all':
            return [transforms.Compose([transforms.Normalize(mean=self.cfg['DATA']['MEAN'],
                                                             std=self.cfg['DATA']['STD'])]),
                    FeatDataset.flip_transform]
        return None

    def get_label_count(self):
        weights = np.zeros(self.num_classes)
        for label in self.labels:
            weights[int(label)] += 1
        return weights

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        feat = self.features[idx]
        if self.transform:
            for tr in self.transform:
                tr(feat[:-1])
        return feat, label


class LinkerAsLabelDS(FeatDataset):
    def __init__(self, cfg, is_train=True):
        FeatDataset.__init__(self, cfg, is_train)
        linker_labels = self.features[:, -1]
        linker_labels[linker_labels < 0.2] = 1
        linker_labels[linker_labels < 1] = 0
        self.labels = linker_labels
        self.features = self.features[:, :-1]

    def __getitem__(self, idx):
        labels = self.labels[idx]
        feat = self.features[idx]
        if self.transform:
            for tr in self.transform:
                tr(feat)
        return feat, labels


# FeatDataset.max_scaling_normalization()
# calc_mean_and_std()
# features = load_np_obj('processed_features_trainset', MLP_PROJ_DIRD)
# FeatDataset.clear_redundant_columns(features)
# features_values_histogram()
# extract_outlier_feature_names()
# check_similar_features()
# check_duplicated_features()