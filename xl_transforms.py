import random
from torch_geometric.transforms import BaseTransform
import graph_dataset
import torch


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


class NoiseTransform(BaseTransform):
    def __init__(self, p=0.5, mean=0, std=0.1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, data):
        if random.random() < self.p:
            self.add_noise(data)
        return data

    def add_noise(self, data):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DistNoiseTransform(NoiseTransform):
    def __init__(self, p=0.5, mean=0, std=0.1):
        super(DistNoiseTransform, self).__init__(p, mean, std)

    def add_noise(self, data):
        noise = torch.randn(data.edge_attr_a.size()) * self.std + self.mean
        data.edge_attr_a += noise
        noise = torch.randn(data.edge_attr_b.size()) * self.std + self.mean
        data.edge_attr_b += noise


class AsaNoiseTransform(NoiseTransform):
    def __init__(self, p=0.5, mean=0, std=0.1):
        super(AsaNoiseTransform, self).__init__(p, mean, std)

    def add_noise(self, data):
        noise = torch.randn(data.x_a[:, graph_dataset.ASA_IDX].size()) * self.std + self.mean
        data.x_a[:, graph_dataset.ASA_IDX] += noise
        noise = torch.randn(data.x_b[:, graph_dataset.ASA_IDX].size()) * self.std + self.mean
        data.x_b[:, graph_dataset.ASA_IDX] += noise
