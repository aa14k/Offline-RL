import numpy as np
from senv import MountainCar
import copy
import scipy as sc
from tqdm.notebook import tqdm

def getdata(env, tt, st=1):
    H, w, sd = env.H, env.width, env.getstate().shape[1]
    succ=np.zeros((H,st,sd+2))# stores state,action,cost
    fail=np.zeros((H,ft:=tt-st,sd+2))
    fc,sc=ft,st
    while sc or fc:
        sac=np.zeros((H,w,sd+2))
        sac[:,:,-2] = np.random.randint(3,size=(H,w))
        s = env.reset()
        for h in range(H):
            sac[h,:,:-2]=s
            c, s = env.step(sac[h,:,-2])
            sac[h,:,-1]=c
        fidx=np.flatnonzero(c)[:fc]
        fail[:,ft-fc:ft-(fc:=fc-fidx.size)]=sac[:,fidx]
        sidx=np.flatnonzero(1-c)[:sc]
        succ[:,st-sc:st-(sc:=sc-sidx.size)]=sac[:,sidx]
    return np.append(succ,fail,axis=1)

class fqi(object):
    def __init__(self, data, d, H):
        self.data = data
        self.d = d
        self.H = H

        self.theta_sq = np.zeros((self.H+1,3,self.d))
        self.theta_log = np.zeros((self.H+1,3,self.d))

    def sigmoid(self, x):
        x[x < -36] = -36
        x[x > -36] = 36
        return 1 / (1 + np.exp(-x))
    
    def get_targets(self, theta, h):
        data = self.data.copy()
        tar = {}

        a, c = data[h,:,-2], data[h,:,-1]

        a0 = np.where(a==0)
        a1 = np.where(a==1)
        a2 = np.where(a==2)

        if h != self.H - 1:

            phi_ = data[h+1,:,:-2]

            inner0 = self.sigmoid(np.matmul(phi_, theta[h+1,0]))
            inner1 = self.sigmoid(np.matmul(phi_, theta[h+1,1]))
            inner2 = self.sigmoid(np.matmul(phi_, theta[h+1,2]))

            v = np.minimum(inner0, inner1, inner2)

            tar[0] = c[a0] + v[a0]
            tar[1] = c[a1] + v[a1]
            tar[2] = c[a2] + v[a2]

        else:
            tar[0] = c[a0]
            tar[1] = c[a1] 
            tar[2] = c[a2] 
        
        return tar


    def ls_loss(self, theta, X, Y):
        return np.sum((self.sigmoid(X @ theta) - Y) ** 2)

    def ls_grad(self, theta, X, Y):
        p = self.sigmoid(X @ theta)
        der = p * (1 - p)
        scalar = 2 * (p - Y) * der
        return scalar.T @ X
    
    def log_loss(self, theta, X, Y):
        p = self.sigmoid(X @ theta)
        return -1.0 * np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    def log_grad(self, theta, X, Y):
        p = self.sigmoid(X @ theta)
        scalar = (p - Y)
        return scalar.T @ X
    
    def ls_solve(self, features, tar, theta0):
        self.sol = sc.optimize.minimize(
            self.ls_loss,
            x0 = theta0,
            args = (features, tar),
            jac = self.ls_grad,
            method = 'bfgs'
        )
        return self.sol.x
    
    def log_solve(self, features, tar, theta0):
        self.sol = sc.optimize.minimize(
            self.log_loss,
            x0 = theta0,
            args = (features, tar),
            jac = self.log_grad,
            method = 'bfgs'
        )
        return self.sol.x

    def run_log(self):
        for h in tqdm(range(self.H-1,-1,-1)):
            tar = self.get_targets(self.theta_log, h)
            for a in range(3):
                idx = np.where(self.data[h,:,-2] == a)
                features = np.squeeze(self.data[h,idx,:-2])
                self.theta_log[h,a] = self.log_solve(features, tar[a], self.theta_log[h+1,a])
        return self.theta_log
    
    def run_ls(self):
        for h in tqdm(range(self.H-1,-1,-1)):
            tar = self.get_targets(self.theta_ls, h)
            for a in range(3):
                idx = np.where(self.data[h,:,-2] == a)
                features = np.squeeze(self.data[h,idx,:-2])
                self.theta_ls[h,a] = self.ls_solve(features, tar[a], self.theta_ls[h+1,a])
        return self.theta_ls





# %%
env = MountainCar(800, 2, 6000)
data = getdata(env, 6000)

# %%
def run_exp(H, trials):    
    env = MountainCar(H, 2, trials)
    data = getdata(env, trials)
    agent = fqi(data, 9, H)
    theta_log = agent.run_log()
    theta_sq = agent.run_sq()
    return [theta_log,theta_sq]

# %%
thetas = run_exp(800, 6000)

# %%
X.shape

# %%
Y = np.squeeze(X)

# %%
Y.shape

# %%
