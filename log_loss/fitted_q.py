from features import LinearFeatureMap
import numpy as np
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
import itertools as iters
from joblib import Parallel, delayed
from scipy.optimize import least_squares
import scipy as sc
import timeit
import copy

class FittedQIteration(object):
    def __init__(self,phi,features,data,horizon,num_trials,gamma):
        self.phi = phi
        self.features = features
        self.data = data
        self.H = horizon
        self.n = num_trials
        self.gamma = gamma
        self.d = len(self.phi.order_list)
        self.theta_sq_ = np.zeros((self.H,3,self.d)) 
        self.theta_sq = np.zeros((self.H,3,self.d))
        self.theta_log = np.zeros((self.H,3,self.d))
        self.theta_log_ = np.zeros((self.H,3,self.d))
        self.A = {}
        self.A_ = {}
        self.get_A()
        self.theta_init_sq = np.zeros(self.d)
        self.theta_init_log = np.zeros(self.d)
        
    
    def get_phi(self,state):
        if self.features == 'poly':
            return self.phi.polynomial_basis(state)
        elif self.features == 'fourier':
            phi = self.phi.fourier_basis(state)
            return phi

    def sigmoid(self,x):
        x[x < -36] = -36
        x[x > 36] = 36
        return 1 / (1 + np.exp(-x))
        
        
        
    
    def get_A(self):
        #print('Getting A')
        data = self.data.copy()
        self.a_idx = {}
        for h in (range(self.H)):
            
            s,a,s_ = data[h][0], data[h][1], data[h][3]
            self.a_idx[h,0] = np.where(a==0)
            self.a_idx[h,1] = np.where(a==1)
            self.a_idx[h,2] = np.where(a==2)
            
            self.x = self.get_phi(s)
            if h != self.H - 1:
                self.x_ = self.get_phi(s_)
            
            self.A[h,0] = self.x[self.a_idx[h,0]]
            self.A[h,1] = self.x[self.a_idx[h,1]]
            self.A[h,2] = self.x[self.a_idx[h,2]]
            
            self.A_[h] = self.x_
            #self.A_[h,1] = self.x_[self.a_idx[h,1]]
            #self.A_[h,2] = self.x_[self.a_idx[h,2]]
    
    def get_targets_sq(self,h):
        data = self.data.copy()
        self.tar_sq = {}
        #for h in (range(self.H - 1, -1, -1)):
            #a, c, s_ = data[h][1], data[h][2], data[h][3]
        a, c = data[h][1], data[h][2]
        
        
        a0 = self.a_idx[h,0]
        a1 = self.a_idx[h,1] 
        a2 = self.a_idx[h,2] 
        
        if h != self.H - 1:
            #phi_ = self.get_phi(s_)
            
            phi_ = self.A_[h]
            q = np.zeros((len(a),3))
            
            inner0 = self.sigmoid(np.matmul(phi_,self.theta_sq_[h+1,0]))
            inner1 = self.sigmoid(np.matmul(phi_,self.theta_sq_[h+1,1]))
            inner2 = self.sigmoid(np.matmul(phi_,self.theta_sq_[h+1,2]))
            
            v = self.gamma * np.minimum(inner0,inner1,inner2)
            
            self.tar_sq[h,0] = c[a0] + v[a0]
            self.tar_sq[h,1] = c[a1] + v[a1]
            self.tar_sq[h,2] = c[a2] + v[a2]
            
        else:
            
            self.tar_sq[h,0] = c[a0] 
            self.tar_sq[h,1] = c[a1] 
            self.tar_sq[h,2] = c[a2] 
                
    def get_targets_log(self,h):
        data = self.data.copy()
        self.tar_log = {}
        #for h in (range(self.H)):
            #a, c, s_ = data[h][1], data[h][2], data[h][3]
            
        a, c = data[h][1], data[h][2]
        
        a0 = self.a_idx[h,0]
        a1 = self.a_idx[h,1] 
        a2 = self.a_idx[h,2]

        #a0 = np.where(a==0)
        #a1 = np.where(a==1)
        #a2 = np.where(a==2)
        
        if h != self.H - 1:
            #phi_ = self.get_phi(s_)
            phi_ = self.A_[h]
            q = np.zeros((len(a),3))
            
            inner0 = self.sigmoid(np.matmul(phi_,self.theta_log_[h+1,0]))
            inner1 = self.sigmoid(np.matmul(phi_,self.theta_log_[h+1,1]))
            inner2 = self.sigmoid(np.matmul(phi_,self.theta_log_[h+1,2]))
            
            v = self.gamma * np.minimum(inner0,inner1,inner2)
            
            self.tar_log[h,0] = c[a0] + v[a0]
            self.tar_log[h,1] = c[a1] + v[a1]
            self.tar_log[h,2] = c[a2] + v[a2]
            
        else:
            
            self.tar_log[h,0] = c[a0] 
            self.tar_log[h,1] = c[a1] 
            self.tar_log[h,2] = c[a2] 
                
            
    def solve_LS(self):
        #self.sol = least_squares(self.func, self.theta_init_sq, args=(self.feature_sq,self.obs_sq))
        self.sol = sc.optimize.minimize(
            self.func1,
            x0 = self.theta_init_sq,
            args=(self.feature_sq,self.obs_sq),
            jac = self.jac1,
            method = 'bfgs'
        )
        return self.sol.x

    def func(self, theta, X, Y):
        # Return residual = fit-observed IGNORE FOR 
        return np.sum( (self.sigmoid(X @ theta) - Y) )

    def func1(self, theta, X, Y):
        # Return residual = fit-observed
        return np.sum((self.sigmoid(X @ theta) - Y) ** 2)

    def jac(self, theta, X, Y):
        p = self.sigmoid(np.matmul(X,theta))
        scalar = 2 * (p - Y) * (p * (1-p))
        return X * scalar.reshape(-1,1)

    def jac1(self, theta, X, Y):
        p = self.sigmoid(np.matmul(X,theta))
        scalar = 2 * (p - Y) * (p * (1-p))
        der = scalar.T @ X
        return der

    def objective_function(self, theta, X, Y):
        p = self.sigmoid(np.matmul(X,theta))
        return -1.0 * np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    def objective_derivative(self,theta, X, Y):
        p = self.sigmoid(X @ theta)
        scalar = p - Y
        der = (scalar.T @ X)
        return der
    
    def logistic_regression(self):
        self.res = sc.optimize.minimize(
            self.objective_function,
            x0 = self.theta_init_log,
            args=(self.feature_log,self.obs_log),
            jac = self.objective_derivative,
            #hess = self.objective_hessian,
            method = 'bfgs'
        )
        return self.res.x

    
    def minimize_log(self):
        for h in (range(self.H-1,-1,-1)):
            for a in range(3):
                self.get_targets_log(h)
                self.feature_log = self.A[h,a]
                self.obs_log = self.tar_log[h,a]
                if h != self.H - 1:
                    self.theta_log_init = self.theta_log_[h+1,a]
                self.theta_log[h,a] = self.logistic_regression()
                self.theta_log_ = self.theta_log
                

    def minimize_sq(self):
        for h in (range(self.H-1,-1,-1)):
            for a in range(3):
                self.get_targets_sq(h)
                self.feature_sq = self.A[h,a]
                self.obs_sq = self.tar_sq[h,a]
                if h != self.H - 1:
                    self.theta_sq_init = self.theta_sq_[h+1,a]
                self.theta_sq[h,a] = self.solve_LS()
                self.theta_sq_ = self.theta_sq
                
            
    
    def update_Q_sq(self):
        self.minimize_sq()
        return self.theta_sq

    def update_Q_log(self):
        self.minimize_log()
        return self.theta_log
    
    def run(self,loss):
        if loss == 1:
            self.update_Q_sq()
            return self.theta_sq
        elif loss == 0:
            self.update_Q_log()
            return self.theta_log
        