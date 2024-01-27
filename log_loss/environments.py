import numpy as np

class DeepSea(object):
    def __init__(self,epLen):
        self.H = epLen
        self.nState = int(self.H**2/2)
        self.nAction = 3
        self.create_idx()
        self.create_cost_function()
    
    def reset(self):
        i = np.random.choice(self.H-1)
        j = np.random.choice(i+1)
        state = self.tuple_to_state[i,j]
        state = 0
        return state,False
    
    def create_idx(self):
        self.state_to_tuple = {}
        self.tuple_to_state = {}
        k = 0
        for i in range(self.H):
            j = 0
            while j < i+1:
                self.state_to_tuple[k] = [i,j]
                self.tuple_to_state[i,j] = k
                j += 1
                k += 1
    
    def create_cost_function(self):
        self.cost_function = {}
        for i in range(self.H):
            for j in range(self.H):
                self.cost_function[i,j] = 0
                if i == self.H-1:
                    self.cost_function[i,j] = 1.0
                if j == self.H-1:
                    self.cost_function[i,j] = 1.0
                    
        self.cost_function[self.H-1,self.H-1] = 0.0
    
    def step(self,state,action):
        done = False
        cost = 0
        #state = self.tuple_to_state[states[0],states[1]]
        state_tuple = self.state_to_tuple[state]
        next_state_tuple = state_tuple.copy()
        if action == 0:
            c = np.random.uniform()
            if c < 1.0:
                next_state_tuple[0] += 1
            else:
                next_state_tuple[0] += 1
                next_state_tuple[1] += 1
                
        elif action == 1:
            next_state_tuple[0] += 1
            next_state_tuple[1] += 1
        else:
            next_state_tuple[1] += 1
        
        if state_tuple[0] == self.H-1:
            next_state = None
            done = True
        elif state_tuple[1] == self.H-1:
            next_state = None
            done = True
        else:
            next_state = self.tuple_to_state[next_state_tuple[0],next_state_tuple[1]]
        if done:
            cost = np.random.binomial(1,p=self.cost_function[state_tuple[0],state_tuple[1]])
            return cost,next_state,done
        else:
            cost = self.cost_function[state_tuple[0],state_tuple[1]]
            return cost,next_state,done
        
    
    

    
    
    
class MountainCar(object):
    def __init__(self,horizon):
        self.horizon = horizon
        self.means = np.array([0,1.0])
        self.reset()
        
    
    
    def reset(self):
        #self.pos = np.random.uniform(low=-1.2,high=0.6)
        #self.vel = np.random.uniform(low=-0.07,high=0.07)
        self.pos = -0.5
        self.vel = 0.0
        self.done = False
        self.h = -1
        return [self.pos,self.vel]
    
    
    def step_broadcast(self, s, action, n, var):
        self.h += 1
        
        pos = s[0,:]
        vel = s[1,:]

        noise = np.random.normal(size=len(pos)) * var
        
        vel = vel + 0.001 * action + -0.0025 * np.cos(3 * pos)
        vel = np.where(vel <= -0.07, -0.07, vel)
        vel = np.where(vel >= 0.07, 0.07, vel)
        
        #vel_top_idx = np.where(vel >= 0.07)
        #vel[vel_bottom_idx] = -0.07  
        #vel[vel_top_idx] = 0.07
        
        cost = np.zeros(n)
        
        pos = pos + vel + noise

        pos = np.where(pos <= -1.2, -1.2, pos)
        pos = np.where(pos >= 0.6, 0.6, pos)
        
        if self.h != self.horizon - 1:
            s_ = np.array([pos,vel])
            return cost, s_
        else:
            cost = np.where(pos >= 0.6, np.random.binomial(n=1,p=self.means[0]), np.random.binomial(n=1,p=self.means[1])) #first arguement is for the successful trajectory 
            s_ = np.array([None] * n)
            return cost, s_
        

    def step_broadcast_eval(self, s, action, n, var):
        self.h += 1
        
        pos = s[0,:]
        vel = s[1,:]

        noise = np.random.normal(size=len(pos)) * var
        
        vel = vel + 0.001 * action + -0.0025 * np.cos(3 * pos)
        vel = np.where(vel <= -0.07, -0.07, vel)
        vel = np.where(vel >= 0.07, 0.07, vel)
        
        #vel_top_idx = np.where(vel >= 0.07)
        #vel[vel_bottom_idx] = -0.07  
        #vel[vel_top_idx] = 0.07
        
        cost = np.zeros(n)
        
        pos = pos + vel + noise

        pos = np.where(pos <= -1.2, -1.2, pos)
        pos = np.where(pos >= 0.6, 0.6, pos)
        
        if self.h != self.horizon - 1:
            s_ = np.array([pos,vel])
            return cost, s_
        else:
            cost = np.where(pos >= 0.6, self.means[0], self.means[1]) #first arguement is for the successful trajectory 
            s_ = np.array([None] * n)
            return cost, s_
    
    def step(self, action):
        self.h += 1
        self.vel = max(min(self.vel + 0.001 * action + -0.0025 * np.cos(3 * self.pos),0.07),-0.07)
        cost = 0
        
        if self.pos > 0.6:
            self.pos = 0.6
            
        else:
            self.pos = max(self.pos + self.vel,-1.2)
            
        if self.pos > 0.6:
            self.pos = 0.6
            
        if self.h == self.horizon-1:
            self.done = True
            if self.pos < -0.52:
                cost = np.random.binomial(1,p=1.0)
            else:
                cost = np.random.binomial(1,p=1.0)
            
        return cost, [self.pos, self.vel], self.h, self.done


class Acrobot(object):
    def __init__(self,horizon):
        self.horizon = horizon
        self.reset()
        
    
    
    def reset(self):
        
        return None


    def step(self, action):
        
            
        return None
    
    
    def step_broadcast(self, s, action, n, var):
        self.h += 1
        
        pos = s[0,:]
        vel = s[1,:]

        noise = np.random.normal(size=len(pos)) * var
        
        vel = vel + 0.001 * action + -0.0025 * np.cos(3 * pos)
        vel = np.where(vel <= -0.07, -0.07, vel)
        vel = np.where(vel >= 0.07, 0.07, vel)
        
        #vel_top_idx = np.where(vel >= 0.07)
        #vel[vel_bottom_idx] = -0.07
        #vel[vel_top_idx] = 0.07
        
        cost = np.zeros(n)
        
        pos = pos + vel + noise

        pos = np.where(pos <= -1.2, -1.2, pos)
        pos = np.where(pos >= 0.6, 0.6, pos)
        
        if self.h != self.horizon - 1:
            s_ = np.array([pos,vel])
            return cost, s_
        else:
            cost = np.where(pos >= 0.6, 0, 1)
            s_ = [None] * n
            return cost, s_
    
    



        
        