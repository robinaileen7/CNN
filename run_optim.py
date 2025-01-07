# Optimize layer weights and sigmoid parameters
import torch.optim as optim
import torch

dtype = torch.float
device = torch.device("cuda:0")
from run_nn import model
import torch.nn as nn
import numpy as np
import config
from sim import sim
import os
import sys

sys.path.append(os.getcwd())
set_seed = 0
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def weighted_MSELoss(y_pred, y_tru, weight):
    return (((weight * (y_pred - y_tru)) ** 2).sum(axis=1)).mean()


class run_optim:
    def __init__(self, X_y_set, J_set, miu_sigma_set, n_param, N_size, X_init, rf):
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.2)
        self.n_param = n_param
        self.N_size = N_size
        self.X_init = X_init
        self.X_train = X_y_set[0]
        self.X_test = X_y_set[1]
        self.y_train = X_y_set[2]
        self.y_test = X_y_set[3]
        self.J_train = J_set[0]
        self.J_test = J_set[1]
        self.miu_train = miu_sigma_set[0]
        self.miu_test = miu_sigma_set[1]
        self.sigma_train = miu_sigma_set[2]
        self.sigma_test = miu_sigma_set[3]
        self.rf = rf
        self.batch_size = round(len(self.X_train) / 2)

    @staticmethod
    def loss_function(x):
        if x == 'MSE':
            return nn.MSELoss()
        elif x == 'cross entropy':
            return nn.CrossEntropyLoss()

    def model_train(self):
        optimizer = self.optimizer
        X_train = self.X_train
        y_train = self.y_train
        J_train = self.J_train
        miu_train = self.miu_train
        sigma_train = self.sigma_train
        iter = 1000

        for i in range(iter):
            optimizer.zero_grad()
            y_pred = model(X_train)
            if i == 0:
                print('The initial loss:')
                loss = weighted_MSELoss(y_pred, y_train, torch.tensor([5, 1]))
                print(loss)

            else:
                bias = model.weight()[0]
                weight = model.weight()[1]
                y_adj = torch.Tensor([])
                unit_weight = []
                ret_vol = []
                for j in range(len(miu_train)):
                    _weight = weight[j][0].tolist()
                    _bias = bias[j].item()
                    _weight.append(_bias / self.rf)
                    _min_max_weight = [(x - np.min(_weight)) / (np.max(_weight) - np.min(_weight)) for x in _weight]
                    _unit_weight = [x / sum(_min_max_weight) for x in _min_max_weight[:-1]]
                    unit_weight.append(_unit_weight)
                    J_shock = [J_train[j][0][0].detach().numpy(), J_train[j][0][1].detach().numpy()]
                    obj = sim(miu=miu_train[j], sigma=sigma_train[j], rho=-0.2, J_apt=J_shock,
                              weight=np.array(_unit_weight), _lambda=5, S_0=400, T=1, t=0,
                              time_step=config.time_step, N=config.N, shock=True)
                    path = obj.exact().mean(axis=0)
                    ret = np.prod(1 + np.diff(path) / path[:-1]) - 1
                    vol = np.std(np.diff(path) / path[:-1]) * config.time_step ** 0.5
                    ret_vol.append([ret, vol])
                    sr_shock = ret / vol
                    sr = (y_train[j][0] / y_train[j][1]).item()
                    y_adj = torch.cat(
                        (y_adj, torch.Tensor([1, 0]).view(1, 2) if sr_shock >= sr else torch.Tensor([0, 1]).view(1, 2)),
                        0)
                loss = weighted_MSELoss(y_pred, y_adj, torch.tensor([5, 1]))
            loss.backward()
            optimizer.step()

        print("Loss for training set is")
        print(loss)
        np.savetxt('weight.txt', unit_weight, delimiter=',')

    def model_test(self):
        run_optim.model_train(self)

        # Placeholder for Testing


if __name__ == "__main__":
    import itertools

    N_size = config.N_
    n_param = config.n_param
    X_init = np.array([config.miu, config.sigma_1, config.sigma_2])
    J_init = np.array([config.J_apt_1, config.J_apt_2])
    X = torch.Tensor([])
    J = torch.Tensor([])
    y = torch.Tensor([])
    miu = []
    sigma = []
    for s in itertools.combinations(np.transpose(X_init), N_size):
        obj_in = np.transpose(s)
        X = torch.cat((X, torch.Tensor(obj_in).view(-1, 1, n_param - 1, N_size)), 0)
        _miu = np.array(obj_in[0])
        _sigma = [np.array(obj_in[1]), np.array(obj_in[2])]
        miu.append(_miu)
        sigma.append(_sigma)
        weight = np.array([1 / len(_miu)] * len(_miu))
        obj = sim(miu=_miu, sigma=_sigma, rho=-0.2, J_apt='NA', weight=weight, _lambda=5, S_0=400, T=1, t=0,
                  time_step=config.time_step, N=config.N, shock=False)
        path = obj.exact().mean(axis=0)
        ret = np.prod(1 + np.diff(path) / path[:-1]) - 1
        vol = np.std(np.diff(path) / path[:-1]) * config.time_step ** 0.5
        y = torch.cat((y, torch.Tensor([ret, vol]).view(1, 2)), 0)
    X.requires_grad = True
    for s in itertools.combinations(np.transpose(J_init), config.N_):
        J = torch.cat((J, torch.Tensor(np.transpose(s)).view(-1, 1, 2, N_size)), 0)
    train_split = round(len(X) * 1)
    X_train, X_test = X[:train_split], X[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]
    J_train, J_test = J[:train_split], J[train_split:]
    miu_train, miu_test = miu[:train_split], miu[train_split:]
    sigma_train, sigma_test = sigma[:train_split], sigma[train_split:]
    X_y_set = X_train, X_test, y_train, y_test
    J_set = J_train, J_test
    miu_sigma_set = miu_train, miu_test, sigma_train, sigma_test

    run_optim(X_y_set, J_set, miu_sigma_set, n_param, N_size, X_init, rf=config.rf).model_test()
