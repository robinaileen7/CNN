import numpy as np
from scipy.stats import bernoulli

time_step = 252
# Numbers of simulation
N = 10000

np.random.seed(10)
H_ = 10
N_ = 10
N_sigma = 2
n_param = N_sigma + 2
rf = 0.05

sigma_1 = np.random.uniform(0.15, 0.4, H_)
sigma_2 = np.random.uniform(0.15, 0.4, H_)
sigma = [sigma_1, sigma_2]

_lambda_ = np.random.uniform(0.01, 0.03, N_sigma)

miu = np.array(np.sum([sigma[i] * _lambda_[i] for i in range(N_sigma)], axis=0) + rf)

J_apt_1 = np.array([np.random.uniform(x - 0.15, 0) for x in miu])
J_apt_2 = np.array([np.random.uniform(x + 0.02, 0) for x in miu])

J_apt = [J_apt_1, J_apt_2]