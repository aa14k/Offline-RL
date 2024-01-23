from environments import MountainCar
import numpy as np
import gc 
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import joblib
import timeit
from fitted_q import FittedQIteration
import pickle


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
        tuples = []
        
        for h in (range(H)):
            a = np.random.choice([-1,0,1],size=num_trials)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            tuples.append([s.T,a+1,cost,np.array(s_).T,h])
            s = s_
        x = np.where(tuples[H-1][2]==0)
        if x[0].shape[0] >= num_success:
            return tuples
            

def evaluate_policy(policy,H, var, theta1, theta2, phi): 
    
    if policy == 'log':
        num_trials = 1
        env = MountainCar(H)
        
        s = np.zeros((2,num_trials))
        s[0,:] = np.ones(num_trials) * - 0.5
        #s[0,:] = np.random.uniform(low = -1.2, high = 0.6, size = num_trials)
        env.reset()
        #costs = []
        for h in (range(H)):
            X = phi.fourier_basis(s.T)
            q = np.zeros((num_trials,3))
            for a in range(3):
                q[:,a] = X @ theta1[h,a]
            a = np.argmin(q, axis=1)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            s = s_
            #costs.append(cost)
    else:
        
        num_trials = 1
        env = MountainCar(H)
        
        s = np.zeros((2,num_trials))
        s[0,:] = np.ones(num_trials) * - 0.5
        env.reset()
        #costs = []
        for h in (range(H)):
            X = phi.fourier_basis(s.T)
            q = np.zeros((num_trials,3))
            for a in range(3):
                q[:,a] = X @ theta2[h,a]
            a = np.argmin(q, axis=1)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            s = s_
            #costs.append(cost)
        
    return cost

def run_experiment(H, num_trials, phi, d, num_success=1, gamma = 1.0):
    tuples = get_data(H, num_trials, num_success)
    features = 'fourier'
    agent = FittedQIteration(phi, features, tuples, H, num_trials, gamma, d)
    del(tuples)
    gc.collect()
    theta1 = agent.update_Q_log()
    theta2 = agent.update_Q_sq()
    del(agent)
    var = 0.0
    cost_log = evaluate_policy('log', H, var, theta1, theta2, phi)
    cost_sq = evaluate_policy('sq', H, var, theta1, theta2, phi)
    gc.collect()
    return [cost_log, cost_sq]



def move_successful_trajectories(tuples, H):
    x = np.where(tuples[-1][2] == 0)
    if len(x) == 1:
        idx = x[0][0]
        for h in range(H):
            s,a,c,s_ = tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx]
            s1,a1,c1,s_1 = tuples[h][0][0], tuples[h][1][0], tuples[h][2][0], tuples[h][3][0]
            tuples[h][0][0], tuples[h][1][0], tuples[h][2][0], tuples[h][3][0] = s,a,c,s_
            tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx] = s1,a1,c1,s_1 
        
    else:
        idx = x[0]
        v = range(len(idx))
        for h in range(H):
            s,a,c,s_ = tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx]
            s1,a1,c1,s_1 = tuples[h][0][v], tuples[h][1][v], tuples[h][2][v], tuples[h][3][v]
            tuples[h][0][v], tuples[h][1][v], tuples[h][2][v], tuples[h][3][v] = s,a,c,s_
            tuples[h][0][idx], tuples[h][1][idx], tuples[h][2][idx], tuples[h][3][idx] = s1,a1,c1,s_1 
            
    return tuples


def truncate_data(tuples, H, num_trials):
    tuples_new = []
    for h in range(H):
        tuples_new.append([tuples[h][0][:num_trials],tuples[h][1][:num_trials],tuples[h][2][:num_trials],tuples[h][3][:num_trials],tuples[h][4]])
    del(tuples)
    gc.collect()
    return tuples_new




def run_experiment_fixed_dataset(H, num_trials, phi, d, file_path, gamma = 1.0):
    with open(file_path + '.pickle', 'rb') as handle:
        tuples = joblib.load(handle)
    
    features = 'fourier'
    agent = FittedQIteration(phi, features, tuples, H, num_trials, gamma, d)
    del(tuples)
    theta1 = agent.update_Q_log()
    theta2 = agent.update_Q_sq()
    del(agent)
    var = 0.0
    cost_log = evaluate_policy('log', H, var, theta1, theta2, phi)
    cost_sq = evaluate_policy('sq', H, var, theta1, theta2, phi)
    gc.collect()
    return [cost_log, cost_sq]




#INGORE BELOW



def evaluate_steps(policy, H, var, theta1, theta2, phi): 
    
    if policy == 'log':
        num_trials = 1
        env = MountainCar(H)
        
        s = np.zeros((2,num_trials))
        s[0,:] = np.ones(num_trials) * - 0.5
        #s[0,:] = np.random.uniform(low = -1.2, high = 0.6, size = num_trials)
        env.reset()
        costs = []
        s_vec = []
        step = H + 1
        for h in (range(H)):
            X = phi.fourier_basis(s.T)
            q = np.zeros((num_trials,3))
            for a in range(3):
                q[:,a] = X @ theta1[h,a]
            a = np.argmin(q, axis=1)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            
            if s[0] >= 0.6:
                step = min(step,h)
                
            costs.append(cost)
            s = s_
            s_vec.append(s)
    else:
        
        num_trials = 1
        env = MountainCar(H)
        step = H + 1
        s = np.zeros((2,num_trials))
        s[0,:] = np.ones(num_trials) * - 0.5
        env.reset()
        costs = []
        s_vec = []
        for h in (range(H)):
            X = phi.fourier_basis(s.T)
            q = np.zeros((num_trials,3))
            for a in range(3):
                q[:,a] = X @ theta2[h,a]
            a = np.argmin(q, axis=1)
            cost, s_ = env.step_broadcast(s, a, num_trials, var)
            
            if s[0] >= 0.6:
                step = min(step,h)
                
            costs.append(cost)
            s = s_
            s_vec.append(s)
        
    return costs, s_vec, step

def fixed_trajectory_loop(data, tuples, phi, runs, H, num_trials, njob):
    c = []
    for i in tqdm(range(len(data))):
        tic = timeit.default_timer()
        if data[i] != num_trials:
            tups = []
            for k in (range(runs)):
                tups.append(truncate_data(tuples[k], H, data[i]))
            x = Parallel(n_jobs= njob)(delayed(run_experiment2)(tups[j], H, data[i],phi) for j in range(runs))
        else:
            x = Parallel(n_jobs = njob)(delayed(run_experiment2)(tuples[j], H, data[i],phi) for j in tqdm(range(runs)))
        toc = timeit.default_timer()
        print('Time: %ss' %(toc-tic))
        c.append(x)
    return c



def run_experiment_horizon(H, num_trials, phi, num_success, gamma = 1.0):
    tuples = get_data(H, num_trials, num_success)
    features = 'fourier'
    agent = FittedQIteration(phi,features,tuples,H,num_trials, gamma)
    theta1 = agent.update_Q_log()
    #theta2 = agent.update_Q_sq()
    theta2 = theta1
    var = 0.0
    cost_log,s_vec_log,step_log = evaluate_steps('log', H, var, theta1, theta2, phi)
    cost_sq,s_vec_sq,step_sq = evaluate_steps('sq', H, var, theta1, theta2, phi)
    
    return [step_log, step_sq]



        

