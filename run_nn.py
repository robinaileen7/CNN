# Run Convolutional Neural Network
import torch

dtype = torch.float
device = torch.device("cuda:0")  # Uncommon this to run on GPU
# device = torch.device("cpu") # Uncommon this to run on CPU
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np

set_seed = 0
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channel = 500
        self.kernel_size = (1, config.N_)
        self.conv = nn.Conv2d(1, self.out_channel, self.kernel_size)
        self.m = nn.Linear(config.n_param - 1, 1)
        self.rf = config.rf

    @staticmethod
    def activation_func(type, x):
        if type == 'tanh':
            return F.tanh(x)
        elif type == 'relu':
            return F.relu(x)
        elif type == 'sigmoid':
            return F.sigmoid(x)

    def forward(self, x):
        # After self.conv(x), _x will output self.out_channel # of weighted sum of
        # ret and sigmas
        _x = self.conv(x)
        _x = torch.flatten(_x, 2)

        # Linear function's weights are like beta_1 ... beta_4 that work as coefficients
        # to weighted sum of ret and sigmas
        # Linear function's bia is like beta_0 that works as intercept
        _x = self.m(_x)

        # Sigmoid function is like logistic regression model that calculates
        # each layer's probability of returning a correct value
        _x = self.activation_func('sigmoid', _x)
        _idx_max = torch.argmax(_x, dim=1)
        idx_max = torch.reshape(_idx_max, (-1,))
        k = model.state_dict()['conv.bias']
        _k = k.repeat(idx_max.shape[0], 1)
        best_bias = _k[range(idx_max.shape[0]), idx_max]
        j = model.state_dict()['conv.weight']
        _j = j.repeat(1, idx_max.shape[0], 1, 1)
        best_weight = _j[idx_max, range(idx_max.shape[0])]
        self.best_bias = best_bias
        self.best_weight = best_weight
        _x = torch.max(_x, dim=1).values
        _y = 1 - _x
        _z = torch.cat((_x, _y), dim=-1)
        return _z

    def weight(self):
        return self.best_bias, self.best_weight


model = Net()