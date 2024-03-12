import numpy as np

class MountainCar(object):# action space: [0,1,2]
    def __init__(self,H,width=1,feat=lambda s:s):
        self.H = H
        self.width=width
        self.feat=feat
        self.reset(width)
    def reset(self,width=None):
        if width == None:
            self.width=width
        #self.pos = np.random.uniform(low=-1.2,high=0.6)
        #self.vel = np.random.uniform(low=-0.07,high=0.07)
        self.pos = np.full(width,-0.5)
        self.vel = np.zeros(width)
        self.h = 0
        self.getstate = lambda: self.feat(np.stack((self.pos,self.vel),axis=1))
        return self.getstate()

    def step(self, a):
        self.h+=1
        self.vel = np.clip(self.vel + 1e-3*(a-1) - 2.5e-3*np.cos(3*self.pos),-7e-2,7e-2)
        self.pos = np.clip(self.pos+self.vel,-1.2,0.6)
        cost = (self.h==self.H)*(self.pos==0.6).astype(np.float64)
        return cost, self.getstate()


# class Acrobot(object):
#     def __init__(self,horizon):
#         self.horizon = horizon
#         self.reset()
    
#     def reset(self):
#         return None

#     def step(self, action):
#         return None
    
#     def step_broadcast(self, s, action, n, var):
#         self.h += 1
#         pos = s[0,:]
#         vel = s[1,:]
#         noise = np.random.normal(size=len(pos)) * var
#         vel = vel + 0.001 * action + -0.0025 * np.cos(3 * pos)
#         vel = np.where(vel <= -0.07, -0.07, vel)
#         vel = np.where(vel >= 0.07, 0.07, vel)
#         #vel_top_idx = np.where(vel >= 0.07)
#         #vel[vel_bottom_idx] = -0.07
#         #vel[vel_top_idx] = 0.07
#         cost = np.zeros(n)
#         pos = pos + vel + noise
#         pos = np.where(pos <= -1.2, -1.2, pos)
#         pos = np.where(pos >= 0.6, 0.6, pos)
#         if self.h != self.horizon - 1:
#             s_ = np.array([pos,vel])
#             return cost, s_
#         else:
#             cost = np.where(pos >= 0.6, 0, 1)
#             s_ = [None] * n
#             return cost, s_
    
    



        
        
