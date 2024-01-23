from features import LinearFeatureMap
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import scipy as sc
import timeit
import matplotlib.pyplot as plt
from utils import get_data, run_experiment_fixed_dataset, move_successful_trajectories, truncate_data
import pickle
import gc
import datetime


num_experiments = 3
spacing = 3000 

data = np.zeros(num_experiments, dtype=int)
data[0] = int(1000)
for i in range(1,num_experiments):
    data[i] = int(spacing * i) 
data = np.flip(data)



H = 600
runs = 8
c = []
num_success = 1
num_trials = data[0]

get_new_data = False
data_start = 0


if get_new_data == True:
    for i in tqdm(range(data_start,runs)):
        tuples = get_data(H, num_trials, num_success)
        moved = move_successful_trajectories(tuples, H)
        del(tuples)
        file_path = 'data/mountain_car/mc_data_traj_' + str(data[0]) + '_'  + str(i) + '.pickle'
        with open(file_path, 'wb') as handle:
            pickle.dump(moved, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del(moved)
        gc.collect()






phi = LinearFeatureMap()
phi.init_fourier_features(2,2)
phi.init_state_normalizers(np.array([0.6,0.07]),np.array([-1.2,-0.07]))
d = int(len(phi.order_list))
print(d)
print('Starting FQI')

file_path = 'data/mountain_car/mc_data_traj_' + str(data[0]) + '_'
file_path_trunc = 'data/mountain_car/truncated_data/' + str(data[0]) + '_trunc_'
for i in range(len(data)):
    tic = timeit.default_timer()
    num_trials = int(data[i])
    print(num_trials)

    for k in range(runs):
        with open(file_path + str(k) + '.pickle', 'rb') as handle:
            tuples = joblib.load(handle)
        tuples = truncate_data(tuples, H, num_trials)
        file_path_trunc = 'data/mountain_car/truncated_data/'
        with open(file_path_trunc + str(num_trials) + '_' + str(k) + '.pickle', 'wb') as handle:
            joblib.dump(tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del(tuples)
    gc.collect()



    x = Parallel(n_jobs=-1)(delayed(run_experiment_fixed_dataset)(H, num_trials, phi, d, file_path_trunc + str(num_trials) + '_' + str(j)) for j in tqdm(range(runs)))
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
plt.title('Performance of FQI vs Size of Dataset over' + str(runs) + ' runs with ' + str(num_success) + 'Successful Trajs.')
plt.savefig('results/mc_plot_' + str(current_time) + '.pdf')
plt.legend()
plt.show()