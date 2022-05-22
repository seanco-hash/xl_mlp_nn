
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layers import GCN, HGPSLPool
import graph_dataset


class HGPSLGNN_NotConnected(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, out_size=1, pool_ratio=0.8, dropout=0.0, sample=False,
                 sparse=True, structure_learn=True, lamb=1.0, cb_head=True, omega_head=True, 
                 theta_head=True, phi_head=True, n_layers=3, angles='regression', ca_head=True):
        super(HGPSLGNN_NotConnected, self).__init__()

        self.is_ca = ca_head
        self.is_cb = cb_head
        self.is_omega = omega_head
        self.is_theta = theta_head
        self.is_phi = phi_head
        self.nhid = n_hidden
        self.num_classes = out_size
        self.dropout_ratio = dropout

        self.HGPSLGNN_a = HGPSLGNN(n_feat, n_hidden // 2, out_size, pool_ratio, dropout, sample, sparse,
                                   structure_learn, lamb, include_linear=False, n_layers=n_layers)
        self.HGPSLGNN_b = HGPSLGNN(n_feat, n_hidden // 2, out_size, pool_ratio, dropout, sample, sparse,
                                   structure_learn, lamb, include_linear=False, n_layers=n_layers)
        self.lin1 = torch.nn.Linear(self.nhid * 2 + 1, self.nhid)  # +1 for linker spacer size
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        if self.is_cb:
            self.cb_head = torch.nn.Linear(self.nhid // 2, self.num_classes)
        if self.is_omega:
            if angles == 'bins':
                self.omega_head = torch.nn.Linear(self.nhid // 2, graph_dataset.OMEGA_BINS)
            else:
                self.omega_head = torch.nn.Linear(self.nhid // 2, graph_dataset.OMEGA_SIZE)
        if self.is_theta:
            if angles == 'bins':
                self.theta_head = torch.nn.Linear(self.nhid // 2, graph_dataset.THETA_BINS * 2)
            else:
                self.theta_head = torch.nn.Linear(self.nhid // 2, graph_dataset.THETA_SIZE)
        if self.is_phi:
            if angles == 'bins':
                self.phi_head = torch.nn.Linear(self.nhid // 2, graph_dataset.PHI_BINS * 2)
            else:
                self.phi_head = torch.nn.Linear(self.nhid // 2, graph_dataset.PHI_SIZE)

    def forward(self, data):
        ca_out, cb_out, omega_out, theta_out, phi_out = None, None, None, None, None
        x_a = self.HGPSLGNN_a(None, data.x_a, data.edge_index_a, data.edge_attr_a, data.x_a_batch)
        x_b = self.HGPSLGNN_b(None, data.x_b, data.edge_index_b, data.edge_attr_b, data.x_b_batch)
        # x = torch.cat((x_a, x_b, torch.reshape(data.linker_size, (data.linker_size.size()[0], 1))), dim=1)
        x = torch.cat((x_a, x_b, data.linker_size), dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.log_softmax(self.lin3(x), dim=-1)
        if self.is_ca:
            ca_out = self.lin3(x)
        if self.is_cb:
            cb_out = self.cb_head(x)
        if self.is_omega:
            omega_out = self.omega_head(x)
        if self.is_theta:
            theta_out = self.theta_head(x)
        if self.is_phi:
            phi_out = self.phi_head(x)
        return ca_out, cb_out, omega_out, theta_out, phi_out


class HGPSLGNN(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, out_size=1, pool_ratio=0.8, dropout=0.0, sample=False,
                 sparse=True, structure_learn=True, lamb=1.0, include_linear=True, n_layers=3):
        super(HGPSLGNN, self).__init__()
        self.num_features = n_feat
        self.nhid = n_hidden
        self.num_classes = out_size
        self.pooling_ratio = pool_ratio
        self.dropout_ratio = dropout
        self.sample = sample
        self.sparse = sparse
        self.sl = structure_learn
        self.lamb = lamb
        self.include_linear = include_linear
        self.convs = []
        self.pools = []
        self.n_layers = n_layers

        self.convs.append(GCNConv(self.num_features, self.nhid))
        for i in range(n_layers - 1):
            self.convs.append(GCN(self.nhid, self.nhid))
            self.pools.append(HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl,
                                        self.lamb))
        self.convs = nn.ModuleList(self.convs)
        self.pools = nn.ModuleList(self.pools)
        if include_linear:
            self.lin1 = torch.nn.Linear(self.nhid * 2 + 1, self.nhid) # +1 for linker spacer size
            self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
            self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is None:
            x, edge_index, batch = x.float(), edge_index, batch
            edge_attr = torch.flatten(edge_attr)
        else:
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
            edge_attr = torch.flatten(data.edge_attr)

        x_i = []
        for i in range(self.n_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_attr))
            if self.pooling_ratio > 0 and i < self.n_layers - 1:
                x, edge_index, edge_attr, batch = self.pools[i](x, edge_index, edge_attr, batch)
            x_i.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
        #
        # x = F.relu(self.conv2(x, edge_index, edge_attr))
        # if self.pooling_ratio > 0:
        #     x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(self.conv3(x, edge_index, edge_attr))
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(self.conv4(x, edge_index, edge_attr))
        # x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(x1) + F.relu(x2) + F.relu(x3) + F.relu(x4)
        x = x_i[0]
        for j in range(1, len(x_i)):
            x += F.relu(x_i[j])
        if self.include_linear:
            x = torch.cat((x, torch.reshape(data.linker, (data.linker.size()[0], 1))), dim=1)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            # x = F.log_softmax(self.lin3(x), dim=-1)
            x = self.lin3(x)
        return x


class SimpleGCN(torch.nn.Module):
    def __init__(self, n_node_feat, n_classes, n_hidden=64):
        super().__init__()
        self.conv1 = GCNConv(n_node_feat, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden // 2)
        self.conv4 = GCNConv(n_hidden // 2, n_hidden // 2)
        self.lin = nn.Linear(n_hidden // 2, n_classes)

    @staticmethod
    def label_node_pool(x, batch, device):
        batch_size = int(torch.max(batch)) + 1
        nodes = torch.zeros((batch_size, x[0].size()[0]), device=device, dtype=torch.double)
        for i in range(batch_size):
            nodes[i] = x[batch == i][-1]
        return nodes

    def forward(self, data, device):
        x, edge_index, batch = data.x.double(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # x = x[-1]
        # x = global_mean_pool(x, batch)
        x = SimpleGCN.label_node_pool(x, batch, device)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
        # return F.log_softmax(x, dim=1)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


def create_model(cfg):
    if cfg['num_classes'] == 2:
        out_size = 1
    else:
        out_size = cfg['num_classes']
    # return SimpleGCN(cfg['node_feat'], out_size)
    if cfg['is_connected']:
        return HGPSLGNN(cfg['node_feat'], out_size=out_size,
                        n_hidden=cfg['layers'], dropout=cfg['dropout'],
                        sample=cfg['sample'], pool_ratio=cfg['pool_ratio'])
    return HGPSLGNN_NotConnected(cfg['node_feat'], out_size=out_size,
                                 n_hidden=cfg['layers'], dropout=cfg['dropout'],
                                 sample=cfg['sample'], pool_ratio=cfg['pool_ratio'],
                                 cb_head=cfg['cb_pred'], omega_head=cfg['omega_pred'],
                                 theta_head=cfg['theta_pred'], phi_head=cfg['phi_pred'],
                                 n_layers=cfg['hidden_layers'], angles=cfg['angles'])
