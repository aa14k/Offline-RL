import numpy as np
from scipy.optimize import minimize as scmin

#data - [(ss,as,cs)]_h, ss.shape=(t,statedim), as.shape=cs.shape=(t)
#A - number of actions
def fqi(data, A, loss='log', gamma=1.0):
    theta = np.zeros((len(data)+1,A,data[0][0].shape[1]))
    l, dl = (llog,dllog) if loss=='log' else (lsq,dlsq)
    vhat = np.zeros(data[0][0].shape[0])
    for h in range(len(data)-1,-1,-1):
        ss, as_, cs = data[h]
        qhat = cs + gamma * vhat
        for a in range(A):
            theta[h,a] = scmin(l, x0=theta[h+1,a],
                               args=(ss[as_==a], qhat[as_==a]),
                               jac=dl, method='L-BFGS-B').x
        vhat = sigmoid(np.min(theta[h]@ss.T,axis=0))# shape (t,) TODO check the dimensions out here
    return theta[:-1]#, targets

def lsq(theta, X, Y):
    return np.sum((sigmoid(X@theta)-Y)**2)

def dlsq(theta, X, Y):
    p = sigmoid(X@theta)
    return 2*(p-Y)*(p*(1-p))@X

def llog(theta, X, Y):
    p = sigmoid(X @ theta)
    return -np.sum(Y*np.log(p) + (1-Y)*np.log(1-p))

def dllog(theta, X, Y):
    return (sigmoid(X@theta)-Y)@X

def sigmoid(x):
        return 1/(1+np.exp(-np.clip(x,-36,36)))
