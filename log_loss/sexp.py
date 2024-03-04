import numpy as np
from senv import MountainCar, CartPole, Acrobot

# tot - total trajectories ; suc - number of successes
def getdata(env, tot, suc=1):
    H, w, sd = env.H, env.width, env.getstate().shape[1]
    Ss = np.zeros((H,tot,sd))
    As, Cs = np.zeros((H,tot)), np.zeros((H,tot))
    fa=tot-suc
    while suc or fa:
        as_ = np.random.randint(3,size=(H,w))
        ss, cs = np.zeros((H,w,sd)), np.zeros((H,w))
        s = env.reset()
        for h in range(H):
            ss[h]=s
            c, s = env.step(as_[h])
            cs[h]=c
        succ=np.flatnonzero(1-c)[:suc]
        fail=np.flatnonzero(c)[:fa]
        idx = np.append(succ,fail)
        sl = np.s_[:,-suc-fa:tot-suc-fa+idx.size]
        Ss[sl],As[sl],Cs[sl]=ss[:,idx],as_[:,idx],cs[:,idx]
        suc-=succ.size ; fa-=fail.size
    return list(zip(Ss,As,Cs))

def eval(env, theta):
    w = env.width ; s=env.reset(1)
    cost=0
    env.reset()
    for h in range(env.H):
        c, s = env.step(np.argmin(theta[h]@s.T))
        cost+=c
    env.reset(w)
    return cost

def exper(envtype, envparams, dataparams):
    A = 2 if envtype is CartPole else 3
    env=envtype(*envparams)
    data=getdata(env, *dataparams)
    thetalog = fqi(data,A,'log')
    thetasq = fqi(data,A,'sq')
    return eval(env,thetalog), eval(env,thetasq)
