from features import LinearFeatureMap
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy as sc
import timeit
import matplotlib.pyplot as plt
from utils import run_experiment 
import datetime


num_experiments = 11
spacing = 5000 

data = np.zeros(num_experiments, dtype=int)
data[0] = int(1000)
for i in range(1,num_experiments):
    data[i] = int(spacing * i)
data = np.flip(data)

H = 600
runs = 48
c = []
num_success = 1

phi = LinearFeatureMap()
phi.init_fourier_features(2,2)
phi.init_state_normalizers(np.array([0.6,0.07]),np.array([-1.2,-0.07]))
d = int(len(phi.order_list))
print(d)
print('Starting FQI')
for i in range(len(data)):
    print (data[i])
    tic = timeit.default_timer()
    num_trials = int(data[i])
    x = Parallel(n_jobs=-1)(delayed(run_experiment)(H, num_trials, phi, d, num_success) for j in tqdm(range(runs)))
    toc = timeit.default_timer()
    print('Time: %ss' %(toc-tic))
    c.append(x)
        
print('Evaluating Policies')
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
current_time = datetime.datetime.now()
np.save('results/c_log_'+ str(current_time) + '.npy', c_log)
np.save('results/c_sq_'+ str(current_time) + '.npy', c_sq)

plt.plot(data, costs_log / runs, label = 'log')
plt.plot(data, costs_sq / runs, label='sq')
plt.xlabel('Number of trajectories')
plt.ylabel('$V(\pi_{FQI})$')
plt.legend()
plt.title('Performance of FQI vs Size of Dataset ')
plt.savefig('results/mc_plot_' + str(current_time) + '.pdf')
plt.legend()
plt.show()