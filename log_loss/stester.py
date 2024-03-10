import numpy as np
import itertools as iters

class LinearFeatureMap(object):
    def __init__(self):
        pass
    
    def init_poly_features(self, state_dim, order):
        self.k = state_dim
        self.c = np.array(list(iters.product(np.arange(order+1), repeat=state_dim)))
        self.len_c = len(self.c)
    
    def polynomial_basis(self, state):
        prod = np.ones(self.len_c)
        
        for j in range(self.k):
            prod[:] = prod[:] * (state[j] ** self.c[:,j])
            
        return prod
    
    def init_state_normalizers(self,maxs,mins):
        self.max = maxs
        self.min = mins
        self.range = (self.max - self.min)
        
    
    def normalize_state(self,state):
        return (state - self.min) / self.range
    
    def init_fourier_features(self, state_dim, order):
        self.order_list = np.array(list(iters.product(np.arange(order+1), repeat=state_dim)))
    
    def fourier_basis(self, state):
        state = self.normalize_state(state)
        order_list = self.order_list
        x = np.zeros((len(state),len(order_list)))
        for i in range(len(order_list)):
            x[:,i] = np.matmul(state, order_list[i])
        x = np.cos(np.pi*x)
        return x


    def fourier_basis_old(self, state):
        state = self.normalize_state(state)
        order_list = self.order_list
        state_new = np.array(state).reshape(1,-1)
        scalars = np.einsum('ij, kj->ik', order_list, state_new) #do a row by row dot product with the state. i = length of order list, j = state dimensions, k = 1
        assert scalars.shape == (len(order_list),1)
        phi = np.cos(np.pi*scalars)
        return phi
    
    def init_radial_basis_features(self, dim, centers, widths):
        self.dim = dim
        self.centers = centers
        self.widths = widths
    
    def radial_basis(self,state):
        x = np.zeros(self.dim)
        x = np.exp(- np.linalg.norm(state - self.centers) ** 2 / (2 * self.widths ** 2))
        return x

### sv
from itertools import product
def makebasis(statedim, order):
    return np.array(list(product(np.arange(order+1), repeat=statedim)))

def fourierfeat(states, basis, min, range):
    return np.cos(np.pi*(states-min)/range@basis.T)

b = makebasis(2,2)
feat=lambda s:fourierfeat(s,b,np.array([-1.2,-7e-2]),
                                np.array([1.8,.14]))
