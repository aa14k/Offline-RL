from environments import MountainCar
from features import LinearFeatureMap
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import timeit
import matplotlib.pyplot as plt
from fitted_q import FittedQIteration
import pickle
import datetime
import scipy as sc
from utils import evaluate_policy
import gc

def get_data(H, num_trials, num_success = 1):
    for x in range(10000):
            
        env = MountainCar(H)
        var = 0.0
        s = np.zeros((2,num_trials))
        s[0,:] = np.ones(num_trials) * - 0.5
        #s[0,:] = np.random.uniform(low = -1.2, high = 0.6, size = num_trials)
        env.reset()
        tuples = []
        
        for h in (range(H)):
            a = np.random.choice([-1,0,1],size=num_trials)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            tuples.append([s.T,a+1,cost,np.array(s_).T,h])
            s = s_
        #x = np.where(tuples[H-1][2]==0.5)
        #print(s)
        x = np.where(tuples[-1][0][:,0] >= 0.6)
        #print(x[0].shape[0])
        if x[0].shape[0] >= num_success:
            return tuples
    print('failed')


def move_successful_trajectories(tuples, H, num_success):
    #x = np.where(tuples[-1][2] == 0.5)
    #print(tuples[-1][0][:,0])
    x = np.where(tuples[-1][0][:,0] >= 0.6)
    if len(x) == 1:
        idx = x[0][0]
        for h in range(H):
            s,a,c,s_ = tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx]
            s1,a1,c1,s_1 = tuples[h][0][0], tuples[h][1][0], tuples[h][2][0], tuples[h][3][0]
            tuples[h][0][0], tuples[h][1][0], tuples[h][2][0], tuples[h][3][0] = s,a,c,s_
            tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx] = s1,a1,c1,s_1 
        
    else:
        idxs = x[0]
        for i in range(len(idx)):
            idx = idxs[i]
            v = i 
            for h in range(H):
                if i < num_success:
                    s,a,c,s_ = tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx]
                    s1,a1,c1,s_1 = tuples[h][0][v], tuples[h][1][v], tuples[h][2][v], tuples[h][3][v]
                    tuples[h][0][v], tuples[h][1][v], tuples[h][2][v], tuples[h][3][v] = s,a,c,s_
                    tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx] = s1,a1,c1,s_1 
                else:
                    v = -1.0 * i
                    s,a,c,s_ = tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx]
                    s1,a1,c1,s_1 = tuples[h][0][v], tuples[h][1][v], tuples[h][2][v], tuples[h][3][v]
                    tuples[h][0][v], tuples[h][1][v], tuples[h][2][v], tuples[h][3][v] = s,a,c,s_
                    tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx] = s1,a1,c1,s_1 

    
    return tuples


def truncate_data(tuples, H, num_trials):
    tuples_new = []
    for h in range(H):
        tuples_new.append([tuples[h][0][:num_trials],tuples[h][1][:num_trials],tuples[h][2][:num_trials],tuples[h][3][:num_trials],tuples[h][4]])
    tuples = []
    return tuples_new


def get_fixed_data(H, data, runs, num_trials, num_success):
    num_trials = num_trials + 5 * num_success #so that we have only num_success many good trajectories in the training set. 
    for i in tqdm(range(runs)):
        file_path = 'data/mountain_car/' + str(i) + '_' + str(max(data))
        tuples = get_data(H, num_trials, num_success)
        tuples = move_successful_trajectories(tuples, H, num_success)
        tuples = truncate_data(tuples,H,max(data))
        with open(file_path + '.pkl', 'wb') as f:
            pickle.dump(tuples, f)
        gc.collect()
            
    
def run_experiment_fixed_dataset(H, file_path, num_trials, phi, gamma = 1.0):
    with open(file_path, 'rb') as f:
        tuples = pickle.load(f)
    tuples = truncate_data(tuples, H, num_trials)
    features = 'fourier'
    agent = FittedQIteration(phi, features, tuples, H, num_trials, gamma)
    theta2 = agent.update_Q_sq()
    theta1 = agent.update_Q_log()
    var = 0.0
    tuples = []
    agent = []
    cost_log = evaluate_policy('log', H, var, theta1, theta2, phi)
    cost_sq = evaluate_policy('sq', H, var, theta1, theta2, phi)
    gc.collect()
    return [cost_log, cost_sq]  



data = [15000,12000,9000,6000,3000,1000]
H = 800
runs = 45
c = []
print('getting data')
num_success = 30
num_trials = 300000
#get_fixed_data(H, data, runs, num_trials, num_success)



phi = LinearFeatureMap()
phi.init_fourier_features(2,2)
phi.init_state_normalizers(np.array([0.6,0.07]),np.array([-1.2,-0.07]))
file_path = 'data/mountain_car/'
num_trials = 30000
for i in (range(len(data))):
    tic = timeit.default_timer()
    x = Parallel(n_jobs=-4)(delayed(run_experiment_fixed_dataset)(H, file_path + str(j) + '_' + str(num_trials) + '.pkl', data[i], phi) for j in tqdm(range(runs)))
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
print(c_log)
print(c_sq)
np.save('results/c_log_'+ str(current_time) + '_H' + '_' + str(H) + '_' + str(num_success), c_log)
np.save('results/c_rwsq_'+ str(current_time) + '_H' + '_' + str(H) + '_' + str(num_success), c_sq)

plt.plot(data, costs_log / runs , label = 'log')
plt.plot(data, costs_sq / runs , label='sq')
plt.xlabel('Number of trajectories')
plt.ylabel('$V(\pi_{FQI})$')
plt.legend()
#plt.title('Performance of FQI vs Size of Dataset over' + str(runs) + ' runs with ' + str(num_success) + 'Successful Trajs.')
plt.savefig('results/mc_plot_' + str(current_time) + '.pdf')