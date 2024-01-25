from features import LinearFeatureMap
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy as sc
import timeit
import matplotlib.pyplot as plt
from utils import  run_experiment
import datetime





data = [30000, 25000, 20000, 15000, 10000, 5000, 1000]
H = 600
runs = 56
c = []

phi = LinearFeatureMap()
phi.init_fourier_features(2,2)
phi.init_state_normalizers(np.array([0.6,0.07]),np.array([-1.2,-0.07]))
print('Running FQI')
for i in tqdm(range(len(data))):
    tic = timeit.default_timer()
    x = Parallel(n_jobs=-3)(delayed(run_experiment)(H, data[i], phi, 1) for j in tqdm(range(runs)))
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

current_time = datetime.datetime.now()
np.save('results/c_log_'+ str(current_time), c_log)
np.save('results/c_sq_'+ str(current_time), c_sq)

plt.plot(data, costs_log / runs, label = 'log')
plt.plot(data, costs_sq / runs, label='sq')
plt.xlabel('Number of trajectories')
plt.ylabel('$V(\pi_{FQI})$')
plt.legend()
plt.title('Performance of FQI vs Size of Dataset over' + str(runs) + ' runs with ' + str(num_success) + 'Successful Trajs.')
plt.savefig('results/mc_plot_' + str(current_time) + '.pdf')
plt.legend()