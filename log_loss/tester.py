from features import LinearFeatureMap
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy as sc
import timeit
import matplotlib.pyplot as plt
from utils import  run_experiment
import datetime




numtrials=5
H = 10
runs = 5
c = []

phi = LinearFeatureMap()
phi.init_fourier_features(2,2)
phi.init_state_normalizers(np.array([0.6,0.07]),np.array([-1.2,-0.07]))

print('Running FQI')
tic = timeit.default_timer()
x =run_experiment(H, numtrials, phi, 1)
print(np.sum(x,axis=0))
toc = timeit.default_timer()
print('Time: %ss' %(toc-tic))
c.append(x)


costs_log = np.zeros(len(data))
costs_sq = np.zeros(len(data))
c_log = np.zeros((len(data),runs))
c_sq = np.zeros((len(data),runs))
for i in range(len(data)):
    for j in range(len(c[i])):
        c_log[i,j] = c[i][j][0][0]
        c_sq[i,j] = c[i][j][1][0]
        costs_log[i] += c[i][j][0][0]
        costs_sq[i] += c[i][j][1][0]

err_log = sc.stats.sem(c_log.T)
err_sq = sc.stats.sem(c_sq.T)
