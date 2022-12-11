
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn.glob
import layers
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.models import MLP
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn.norm.graph_norm import GraphNorm
from layers import GCN, HGPSLPool, CaPool
import graph_dataset


class GeneralGNN(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, out_size=1, pool_ratio=0.8, dropout=0.0, sample=False,
                 sparse=True, structure_learn=True, lamb=1.0, cb_head=True, omega_head=True,
                 theta_head=True, phi_head=True, n_layers=3, angles='regression', ca_head=True,
                 loss_type='bce', gnn_type='gcn', act_func='relu', batchnorm=False, n_attn_heads=2,
                 is_xl_type=False, is_spacer_size=True, pool_func='sum', is_ca_pool=False, attn_dropout=0.0):
        super(GeneralGNN, self).__init__()

        self.is_ca = ca_head
        self.is_cb = cb_head
        self.is_omega = omega_head
        self.is_theta = theta_head
        self.is_phi = phi_head
        self.nhid = n_hidden
        self.dropout = nn.Dropout(dropout)
        self.is_xl_type = is_xl_type
        self.is_spacer_size = is_spacer_size
        self.act = GeneralGNN.get_activation_func(act_func)
        self.is_ordinal = loss_type == 'ordinal'
        self.pool_func = layers.customPool(n_hidden, pool_ratio, sample, sparse, structure_learn, lamb, pool_func,
                                           is_ca_pool)
        if loss_type == 'corn':
            self.num_classes = out_size - 1
        else:
            self.num_classes = out_size

        self.GNN_A, _ = GeneralGNN.get_wraped_gnn(n_feat, n_hidden, out_size, pool_ratio, dropout, sample, sparse,
                                               structure_learn, lamb, n_layers, gnn_type, self.act, batchnorm,
                                                  n_attn_heads, self.pool_func, is_ca_pool, attn_dropout)
        self.GNN_B, x_dim = GeneralGNN.get_wraped_gnn(n_feat, n_hidden, out_size, pool_ratio, dropout, sample, sparse,
                                               structure_learn, lamb, n_layers, gnn_type, self.act, batchnorm,
                                                      n_attn_heads, self.pool_func, is_ca_pool, attn_dropout)
        if pool_func == 'hgpsl':
            x_dim *= 2
        add_dims = 0
        if self.is_spacer_size:
            add_dims += 1
        if self.is_xl_type:
            add_dims += 2
        # self.lin1 = torch.nn.Linear(2 * x_dim + add_dims, self.nhid)
        self.lin1 = torch.nn.Linear(2 * x_dim, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid + add_dims, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        if self.is_ordinal:
            self.sig = torch.nn.Sigmoid()
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

    @staticmethod
    def get_activation_func(act_func):
        if act_func == 'relu':
            return nn.ReLU()
        elif act_func == 'prelu':
            return nn.PReLU()

    @staticmethod
    def get_wraped_gnn(n_feat, n_hidden=128, out_size=1, pool_ratio=0.8, dropout=0.0, sample=False,
                       sparse=True, structure_learn=True, lamb=1.0, n_layers=3, gnn_type='gcn', act=nn.ReLU(),
                       batchnorm=False, n_attn_heads=2, pool_func=layers.customPool(), is_ca_pool=False, attn_dropout=0.0):
        x_dim = 0
        if gnn_type == 'gine':
            n = CustomGIN(n_feat, n_hidden, dropout, n_layers, act, True, batchnorm)
            x_dim = n_hidden * n_layers
        elif gnn_type == 'gin':
            n = CustomGIN(n_feat, n_hidden, dropout, n_layers, act, False, batchnorm)
            x_dim = n_hidden * n_layers
        elif gnn_type == 'gcn':
            n = CustomGCN(n_feat, n_hidden, dropout, n_layers, act, batchnorm)
            x_dim = n_hidden * n_layers * 2
        elif gnn_type == 'hgpsl':
            n = HGPSLGNN(n_feat, n_hidden, out_size, pool_ratio, dropout, sample, sparse,
                         structure_learn, lamb, include_linear=False, n_layers=n_layers)
            x_dim = n_hidden * 2 * n_layers
        elif gnn_type == 'gat':
            n = CustomGAT(n_feat, n_hidden, dropout, n_layers, act, n_attn_heads, batchnorm, dropout, attn_dropout, pool_func)
            x_dim = n_hidden * n_layers
        if is_ca_pool:
            x_dim += (n_hidden * n_layers)
        return n, x_dim

    def forward(self, data):
        ca_out, cb_out, omega_out, theta_out, phi_out = None, None, None, None, None
        x_a = self.GNN_A(None, data.x_a, data.edge_index_a, data.edge_attr_a, data.x_a_batch)
        x_b = self.GNN_B(None, data.x_b, data.edge_index_b, data.edge_attr_b, data.x_b_batch)
        x = torch.cat((x_a, x_b), dim=1)
        # if self.is_spacer_size:
        #     x = torch.cat((x, data.linker_size), dim=1)
        # if self.is_xl_type:
        #     x = torch.cat((x, data.xl_type), dim=1)
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        if self.is_spacer_size:
            x = torch.cat((x, data.linker_size), dim=1)
        if self.is_xl_type:
            x = torch.cat((x, data.xl_type), dim=1)
        x = self.act(self.lin2(x))
        x = self.dropout(x)
        if self.is_ca:
            ca_out = self.lin3(x)
            if self.is_ordinal:
                ca_out = self.sig(ca_out)
        if self.is_cb:
            cb_out = self.cb_head(x)
            if self.is_ordinal:
                cb_out = self.sig(cb_out)
        if self.is_omega:
            omega_out = self.omega_head(x)
        if self.is_theta:
            theta_out = self.theta_head(x)
        if self.is_phi:
            phi_out = self.phi_head(x)
        return ca_out, cb_out, omega_out, theta_out, phi_out


class HGPSLGNN_NotConnected(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, out_size=1, pool_ratio=0.8, dropout=0.0, sample=False,
                 sparse=True, structure_learn=True, lamb=1.0, cb_head=True, omega_head=True, 
                 theta_head=True, phi_head=True, n_layers=3, angles='regression', ca_head=True,
                 loss_type='bce'):
        super(HGPSLGNN_NotConnected, self).__init__()

        self.is_ca = ca_head
        self.is_cb = cb_head
        self.is_omega = omega_head
        self.is_theta = theta_head
        self.is_phi = phi_head
        self.nhid = n_hidden
        self.dropout_ratio = dropout
        self.is_ordinal = loss_type == 'ordinal'
        if loss_type == 'corn':
            self.num_classes = out_size - 1
        else:
            self.num_classes = out_size

        self.HGPSLGNN_a = HGPSLGNN(n_feat, n_hidden // 2, out_size, pool_ratio, dropout, sample, sparse,
                                   structure_learn, lamb, include_linear=False, n_layers=n_layers)
        self.HGPSLGNN_b = HGPSLGNN(n_feat, n_hidden // 2, out_size, pool_ratio, dropout, sample, sparse,
                                   structure_learn, lamb, include_linear=False, n_layers=n_layers)
        self.lin1 = torch.nn.Linear(self.nhid * 2 + 1, self.nhid)  # +1 for linker spacer size
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        if self.is_ordinal:
            self.sig = torch.nn.Sigmoid()
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
        x = torch.cat((x_a, x_b, data.linker_size), dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        if self.is_ca:
            ca_out = self.lin3(x)
            if self.is_ordinal:
                ca_out = self.sig(ca_out)
        if self.is_cb:
            cb_out = self.cb_head(x)
            if self.is_ordinal:
                cb_out = self.sig(cb_out)
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
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.n_layers = n_layers

        self.pre_lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.pre_lin2 = torch.nn.Linear(self.nhid, self.nhid)

        self.convs.append(GCN(self.nhid, self.nhid))
        for i in range(n_layers - 1):
            self.convs.append(GCN(self.nhid, self.nhid))
            self.pools.append(HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl,
                                        self.lamb))
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

        x = F.relu(self.pre_lin1(x))
        x = F.relu(self.pre_lin2(x))

        x_i = []
        for i in range(self.n_layers):
            x = F.relu(self.convs[i](x, edge_index, edge_attr))
            if self.pooling_ratio > 0 and i < self.n_layers - 1:
                x, edge_index, edge_attr, batch = self.pools[i](x, edge_index, edge_attr, batch)
            x_i.append(torch.cat([gsp(x, batch), gap(x, batch)], dim=1))
        # x = x_i[0]
        # for j in range(1, len(x_i)):
        #     x += F.relu(x_i[j])
        x = torch.cat(x_i, dim=1)
        if self.include_linear:
            x = torch.cat((x, torch.reshape(data.linker, (data.linker.size()[0], 1))), dim=1)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            x = self.lin3(x)
        return x


class CustomGAT(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, dropout=0.0, n_layers=3, act=nn.ReLU(), n_heads=2, batchnorm=False,
                 in_dropout=0.0, attn_dropout=0.0, pool_func=layers.customPool()):
        super(CustomGAT, self).__init__()
        self.nhid = n_hidden
        self.num_features = n_feat
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(in_dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = act
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.batchnorm = batchnorm
        self.pool_func = pool_func
        self.pre_lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.pre_lin2 = torch.nn.Linear(self.nhid, self.nhid)
        if n_heads % self.nhid != 0:
            n_heads = 8
        # self.convs.append(GCNConv(self.nhid, self.nhid, add_self_loops=False, improved=True))
        # self.convs.append(GATv2Conv(self.nhid, self.nhid // n_heads, heads=n_heads, dropout=attn_dropout, edge_dim=1,
        #                             add_self_loops=False))
        # self.convs.append(GATv2Conv(self.nhid, self.nhid // n_heads, heads=n_heads, dropout=attn_dropout, edge_dim=1,
        #                             add_self_loops=False))
        for i in range(n_layers):
            self.convs.append(GATv2Conv(self.nhid, self.nhid // n_heads, heads=n_heads, dropout=attn_dropout, edge_dim=1,
                                        add_self_loops=False))
            if batchnorm:
                # self.norms.append(nn.BatchNorm1d(self.nhid))
                # self.norms.append(BatchNorm(self.nhid))
                self.norms.append(GraphNorm(self.nhid))

    def forward(self, data, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is None:
            x, edge_index, batch = x.float(), edge_index, batch
            edge_attr = torch.flatten(edge_attr)
        else:
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
            edge_attr = torch.flatten(data.edge_attr)

        x = self.act(self.pre_lin1(x))
        x = self.input_dropout(self.act(self.pre_lin2(x)))
        x_i = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            if self.batchnorm:
                x = self.norms[i](x)
            x = self.act(x)
            x, edge_index, edge_attr, batch, x_pool = self.pool_func(x, edge_index, edge_attr, batch)
            x_i.append(x_pool)
            # x = self.dropout(x)
        x = torch.cat(x_i, dim=1)
        return x


class CustomGCN(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, dropout=0.0, n_layers=3, act=nn.ReLU(), batchnorm=False):
        super(CustomGCN, self).__init__()
        self.nhid = n_hidden
        self.num_features = n_feat
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.batchnorm = batchnorm
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.n_layers = n_layers

        self.pre_lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.pre_lin2 = torch.nn.Linear(self.nhid, self.nhid)

        for i in range(n_layers):
            self.convs.append(GCNConv(self.nhid, self.nhid))
            self.linears.append(torch.nn.Linear(self.nhid, self.nhid))
            if batchnorm:
                self.norms.append(nn.BatchNorm1d(self.nhid))

    def forward(self, data, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is None:
            x, edge_index, batch = x.float(), edge_index, batch
            edge_attr = torch.flatten(edge_attr)
        else:
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
            edge_attr = torch.flatten(data.edge_attr)

        x = self.act(self.pre_lin1(x))
        x = self.act(self.pre_lin2(x))
        x_i = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            if self.batchnorm:
                x = self.norms[i](x)
            x = self.act(x)
            x_i.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
        x = torch.cat(x_i, dim=1)
        return x


class CustomGIN(torch.nn.Module):
    def __init__(self, n_feat, n_hidden=128, dropout=0.0, n_layers=3, act=nn.ReLU(), is_gine=False, batchnorm=False):
        super(CustomGIN, self).__init__()
        self.nhid = n_hidden
        self.num_features = n_feat
        self.gins = nn.ModuleList()
        self.act = act
        self.n_layers = n_layers
        self.is_gine = is_gine

        self.pre_lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.pre_lin2 = torch.nn.Linear(self.nhid, self.nhid)

        b_norm = None
        if batchnorm:
            b_norm = nn.BatchNorm1d(self.nhid)

        for i in range(n_layers):
            if is_gine:
                self.gins.append(GINEConv(MLP([self.nhid, self.nhid, self.nhid], act=self.act, batch_norm=batchnorm),
                                          eps=0.1, train_eps=True, edge_dim=1))
            else:
                self.gins.append(GINConv(MLP([self.nhid, self.nhid, self.nhid],  act=self.act, batch_norm=batchnorm),
                                         eps=0.1, train_eps=True))

    def forward(self, data, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is None:
            x, edge_index, batch = x.float(), edge_index, batch
        else:
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        if self.is_gine:
            edge_attr = edge_attr.reshape((edge_attr.shape[0], 1))
        x = self.act(self.pre_lin1(x))
        x = self.act(self.pre_lin2(x))
        x_i = []
        for i in range(self.n_layers):
            if self.is_gine:
                x = self.gins[i](x, edge_index, edge_attr)
            else:
                x = self.gins[i](x, edge_index)
            x_i.append(gsp(x, batch))
        x = torch.cat(x_i, dim=1)
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
    return GeneralGNN(cfg['node_feat'], out_size=out_size,
                      n_hidden=cfg['layers'], dropout=cfg['dropout'],
                      sample=cfg['sample'], pool_ratio=cfg['pool_ratio'],
                      cb_head=cfg['cb_pred'], omega_head=cfg['omega_pred'],
                      theta_head=cfg['theta_pred'], phi_head=cfg['phi_pred'],
                      n_layers=cfg['hidden_layers'], angles=cfg['angles'],
                      loss_type=cfg['loss_type'], gnn_type=cfg['gnn_type'], act_func=cfg['activation'],
                      batchnorm=cfg['batchnorm'], n_attn_heads=cfg['attention_heads'], is_xl_type=cfg['xl_type_feature'],
                      is_spacer_size=cfg['spacer_feture'], pool_func=cfg['global_pool_func'],
                      is_ca_pool=cfg['is_ca_pool'], attn_dropout=cfg['attn_dropout'])
