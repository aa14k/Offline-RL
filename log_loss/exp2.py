from features import LinearFeatureMap
from environments import MountainCar
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy as sc
import timeit
import matplotlib.pyplot as plt
from fitted_q_numpy import FittedQIteration
from utils import evaluate_policy
import gc
import datetime 

def get_data(H, num_trials, num_success=None):
    for x in range(5000):
        
        if num_success == None:
            num_success = 1
            
        env = MountainCar(H)
        var = 0.0
        s = np.zeros((2,num_trials))
        s[0,:] = np.ones(num_trials) * - 0.5
        #s[0,:] = np.random.uniform(low = -1.2, high = 0.6, size = num_trials)
        env.reset()
        tuples = np.zeros((H,num_trials,6))
        
        for h in (range(H)):
            a = np.random.choice([-1,0,1],size=num_trials)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            #tuples.append([s.T,a+1,cost,np.array(s_).T,h])
            tuples[h,:,0] = s[0,:]
            tuples[h,:,1] = s[1,:]
            tuples[h,:,2] = a
            tuples[h,:,3] = cost
            if h != H-1:
                tuples[h,:,4] = s_[0]
                tuples[h,:,5] = s_[1]
            else:
                tuples[h,:,4] = -99999
                tuples[h,:,5] = 99999

            
            s = s_
        x = np.where(tuples[H-1,:,3] ==0 )
        if x[0].shape[0] >= num_success:
            return tuples


def move_successful_trajectories(tuples, H):
    x = np.where(tuples[-1,:,3] == 0)    
    idx = x[0]
    v = range(len(idx))
    temp = tuples.copy()
    tuples[:,v,:] = temp[:,idx,:]
    tuples[:,idx,:] = temp[:,v,:]

    temp = []
    
    return tuples


def run_experiment_fixed_dataset(H, num_trials, phi, d, file_path, gamma = 1.0):
    data = np.load(file_path)
    tuples = data[:,0:num_trials,:]
    data = []
    features = 'fourier'
    agent = FittedQIteration(phi, features, tuples, H, num_trials, gamma, d)
    tuples = []
    theta1 = agent.update_Q_log()
    theta2 = agent.update_Q_sq()
    agent = []
    var = 0.0
    cost_log = evaluate_policy('log', H, var, theta1, theta2, phi)
    cost_sq = evaluate_policy('sq', H, var, theta1, theta2, phi)
    gc.collect()
    return [cost_log, cost_sq]


data = [30000, 25000, 20000, 15000, 10000, 5000, 1000]
runs = 96
num_traj = max(data)
num_success = 1
H = 800
new_data = True

if new_data == True:
    #tuples = np.zeros(( H, num_traj, 6))
    for i in tqdm(range(runs)):
        tuples = get_data(H, num_traj, num_success)
        tuples = move_successful_trajectories(tuples, H)
        np.save('data/mountain_car/tuples_' + str(num_traj) + '_' + str(i), tuples) 

tuples = []

phi = LinearFeatureMap()
phi.init_fourier_features(2,2)
phi.init_state_normalizers(np.array([0.6,0.07]),np.array([-1.2,-0.07]))
d = int(len(phi.order_list))
print(d)
print('Starting FQI')

c = []
cpu_cores = -1

for i in range(len(data)):
    tic = timeit.default_timer()
    num_trials = int(data[i])

    file_path = 'data/mountain_car/tuples_' + str(num_traj) + '_'
    print('loop')
    x = Parallel(n_jobs=cpu_cores)(delayed(run_experiment_fixed_dataset)(H, num_trials, phi, d, file_path + str(j) + '.npy') for j in tqdm(range(runs)))
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