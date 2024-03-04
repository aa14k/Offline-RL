import numpy as np
import scipy as sc
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression


class Sigmoid_Regression(object):
    def __init__(self, features, obs, d, theta_star, lr_tol = pow(10,-8)):
        self.features = features
        self.obs = obs
        self.n = len(obs)
        self.d = d
        self.theta_star = theta_star
        self.logit_regression_tol = lr_tol
        theta_init = np.random.normal(size=self.d)
        
        self.theta_ls = theta_init
        self.theta = theta_init
        
 
    def solve_LS(self):
        #Solve nonlinear least squares using scipy black-box (since sq loss is non-convex wrt to theta)
        #A = np.zeros((self.d,self.d))
        #b = np.zeros(self.d)
        #for i in range(self.n):
        #    x, y = self.features[i], self.obs[i]
        #    A = A + np.outer(x, x)
        #    b = b + y * x
        #self.theta_ols = np.linalg.solve(A,b)
        sol = sc.optimize.least_squares(self.func, self.theta_ls, args=(self.features,self.obs))
        self.theta_ls = sol.x
    
    def solve_Logit_Regression(self):
        # note pretty sure that setting random state here does nothing
        self.clf = LogisticRegression(tol = self.logit_regression_tol, random_state = 0).fit(self.features, self.obs)
        return np.array(self.clf.coef_)



    def sigmoid(self, x):
        x[x < -36] = -36
        x[x > 36] = 36
        return 1 / (1 + np.exp(-x))
        
    def objective_function(self, theta, X, Y):
        p = self.sigmoid(X @ theta)
        return -1.0 * np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    
    
    def logistic_regression(self):
        self.res = sc.optimize.minimize(
            self.objective_function,
            x0 = np.zeros(self.d),
            args=(self.features,self.obs),
            tol = pow(10,-8)
        )
        return self.res.x
        
    
    def get_grad_hessian_log(self,theta):
        
        inner = np.dot(self.features, self.theta)
        grad = np.zeros(self.d)
        X = self.features.copy()
        
        mu = np.zeros((self.n,self.d))
        mu[:,0] = -1.0 * (2 * self.obs - np.tanh(inner) - 1)
        dot_mu_root = np.sqrt(1 / np.cosh(inner))
        
        
        for i in range(self.d):
            if i != 0:
                mu[:,i] = mu[:,0]
            X[:,i] = np.multiply(X[:,i], dot_mu_root)
            
        mult = np.multiply(mu,self.features)
        grad = np.sum(mult, axis = 0)
        
        hess = np.einsum('ij,ik->jk', X, X)
        
        return hess, grad
    
   
    def p(self,x):
        x = np.where(x >= 500, 500, x)
        x = np.where(x <= -500, -500, x)
        return 1 / (1 + np.exp(-x))
    
    def func(self, theta, x, y):
        # Return residual = fit-observed
        return self.p(np.inner(x,theta)) - y
        

    
    def Newton_Method(self):
        
        print(np.linalg.norm(self.theta_ls - self.theta_star))
        
        #Solve logisitic regression using newton's method (since log loss is convex wrt to theta)
        for i in tqdm(range(self.newton_iters)):
            
            hess, grad = self.get_grad_hessian_log(self.theta)
            #hess_ls, grad_ls = self.get_grad_hessian_sq(self.theta_ls)
            
            newton_step = np.linalg.solve(hess, grad)
            #newton_step_ls = np.linalg.solve(hess_ls, grad_ls)
            
            self.theta = self.theta - newton_step
            #self.theta_ls = self.theta_ls - newton_step_ls
            
            if i % 1000 == 999:
                print(np.linalg.norm(newton_step))
