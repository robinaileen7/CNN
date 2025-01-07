# Simulate log sum of weighted assets
import numpy as np
import math
from scipy.stats import bernoulli
import config


class sim:
    def __init__(self, miu, sigma, rho, J_apt, weight, _lambda, S_0, T, t, time_step, N, shock):
        self.miu = miu
        # sigma as an array
        self.sigma = sigma
        self.rho = rho
        self.J_apt = J_apt
        self.weight = weight
        self._lambda = _lambda
        self.S_0 = S_0
        self.T = T
        self.t = t
        self.time_step = time_step
        self.N = N
        self.shock = shock
        self.rf = 0.005

    def dt(self):
        return 1 / self.time_step

    @staticmethod
    def drift(miu, sigma, rho, weight, rf):
        return np.sum(miu * weight) + rf * (1 - np.sum(weight)) - 0.5 * (
                    np.sum(weight * sigma[0] ** 2) + np.sum(weight * sigma[1] ** 2) + 2 * np.sum(
                rho * weight * sigma[0] * sigma[1]))

    @staticmethod
    def diffusion_sum(sigma, rho, weight, n_path, N):
        W_1 = np.random.normal(0, 1, (n_path, N))
        Z_2 = np.random.normal(0, 1, (n_path, N))
        W_2 = np.array([rho * x + np.sqrt(1 - rho ** 2) * y for x, y in zip(W_1, Z_2)])
        return np.sum(weight * sigma[0]) * W_1 + np.sum(weight * sigma[1]) * W_2

    def exact(self):
        miu = self.miu
        sigma = self.sigma
        rho = self.rho
        J_apt = self.J_apt
        weight = self.weight
        _lambda = self._lambda
        S_0 = self.S_0
        T = self.T
        t = self.t
        time_step = self.time_step
        N = self.N
        shock = self.shock
        rf = self.rf

        n_path = round((T - t) * time_step) + 1
        S_array = np.zeros((N, n_path))
        S_array[:, 0] = S_0
        drift = sim.drift(miu, sigma, rho, weight, rf)
        np.random.seed(10)
        diffusion_sum = sim.diffusion_sum(sigma, rho, weight, n_path, N)
        N_t = np.zeros((n_path, N))
        J_sum = np.zeros((n_path, N))
        if shock:
            np.random.seed(1)
            N_t = bernoulli.rvs(size=(n_path, N), p=1 - np.exp(-_lambda * self.dt()))
            np.random.seed(5)
            J_sum = np.sum([z * np.random.uniform(x, y, (n_path, N)) for x, y, z in zip(J_apt[0], J_apt[1], weight)],
                           axis=0)
        for i in range(1, n_path):
            S_array[:, i] = S_array[:, i - 1] + 500 * (
                        drift * self.dt() + diffusion_sum[i] * self.dt() ** (1 / 2) + N_t[i] * J_sum[i])
        return S_array


if __name__ == "__main__":
    time_step = 252
    N = 10000
    miu = config.miu
    sigma = [config.sigma_1, config.sigma_2]
    J_apt = [config.J_apt_1, config.J_apt_2]
    obj = sim(miu=miu, sigma=sigma, rho=-0.2, J_apt=J_apt, weight=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
              _lambda=5, S_0=400, T=1, t=0, time_step=time_step, N=N, shock=False)
    path = obj.exact().mean(axis=0)
    ret = np.prod(1 + np.diff(path) / path[:-1]) - 1
    vol = np.std(np.diff(path) / path[:-1]) * time_step ** 0.5
    print(ret, vol)