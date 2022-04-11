
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data_proccess import load_np_obj
from data_proccess import MLP_PROJ_DIRD
import ResMLP


class RayMLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''
    def __init__(self, l1=224, l2=112, l3=56):
        super().__init__()
        self.input_size = 397
        self.output_size = 1
        self.drop_rate = 0.5
        h_sizes = [self.input_size, l1, l2, l3]

        self.hiddens = nn.ModuleList()
        # self.hiddens.append(rel1)
        for i in range(len(h_sizes) - 1):
            self.hiddens.append(nn.Linear(h_sizes[i], h_sizes[i+1]))
            self.hiddens.append(nn.ReLU())
            self.hiddens.append(nn.Dropout(self.drop_rate))
        self.head = nn.Linear(l3, self.output_size)

    def forward(self, x):
        '''Forward pass'''
        for layer in self.hiddens:
            x = layer(x)
        x = self.head(x)
        return x


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''
    def __init__(self, input_size=397, output_size=3, drop_rate=0.5, n_hidden=2, initial_hidden_size=256,
                 cfg=None):
        super().__init__()
        if n_hidden == 0:
            initial_hidden_size = output_size
        prev_size = input_size
        next_size = initial_hidden_size
        self.hiddens = nn.ModuleList()
        if cfg['MODEL']['LAYERS']:
            layers = [input_size]
            layers += cfg['MODEL']['LAYERS']
            for i in range(len(layers) - 1):
                self.hiddens.append(nn.Linear(layers[i], layers[i + 1]))
                self.hiddens.append(nn.ReLU())
                if cfg['MODEL']['BATCHNORM'] and i < len(layers) - 2:
                    self.hiddens.append(nn.BatchNorm1d(layers[i + 1]))
                self.hiddens.append(nn.Dropout(drop_rate))
            self.head = nn.Linear(layers[-1], output_size)
        else:
            for i in range(n_hidden):
                if prev_size > 16:
                    next_size = prev_size // 2
                self.hiddens.append(nn.Linear(prev_size, next_size))
                self.hiddens.append(nn.ReLU())
                self.hiddens.append(nn.Dropout(drop_rate))
                prev_size = next_size
            if n_hidden == 0:
                self.head = nn.Identity()
            else:
                self.head = nn.Linear(prev_size, output_size)

    def forward(self, x):
        '''Forward pass'''
        # x = self.layers(x)
        # x = self.input_layer(x)
        for layer in self.hiddens:
            x = layer(x)
        x = self.head(x)
        return x


def create_model(cfg, ray_config=None):
    if cfg['TRAIN']['HPARAM_METHOD'] == 'ray':
        model = RayMLP(ray_config["l1"], ray_config["l2"], ray_config["l3"])
        return model
    drop_rate = cfg['MODEL']['DROPOUT']
    input_size = cfg['MODEL']['INPUT_SIZE']
    output_size = cfg['MODEL']['OUTPUT_SIZE']
    n_hidden_layers = cfg['MODEL']['HIDDEN_LAYERS']
    if cfg['MODEL']['TYPE'] == 'resmlp':
        model = ResMLP.ResMLP(drop_rate, 1, n_hidden_layers, output_size, input_size)
    else:
        model = MLP(input_size, output_size, drop_rate, n_hidden_layers, cfg=cfg)
    return model


def get_data(batch_size=32):
    labels = load_np_obj('best_xl_label_trainset', MLP_PROJ_DIRD)
    features = load_np_obj('best_xl_feat_trainset', MLP_PROJ_DIRD)
    tensor_x = torch.Tensor(features)  # transform to torch tensor
    tensor_y = torch.Tensor(labels)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader
    return my_dataloader
