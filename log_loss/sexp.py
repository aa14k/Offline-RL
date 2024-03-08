import numpy as np
from senv import MountainCar, CartPole, Acrobot
from sfqi import fqi

# tot - total trajectories ; suc - number of successes
# todo this is geared to MountainCar
def getdata(env, tot, suc=1):
    H, w, sd = env.H, env.width, env.getstate().shape[1]
    data=np.zeros((H,tot,sd+2))# stores state,action,cost
    # Ss = np.zeros((H,tot,sd))
    # As, Cs = np.zeros((H,tot)), np.zeros((H,tot))
    fa=tot-suc
    while suc or fa:
        # as_ = np.random.randint(3,size=(H,w))
        # ss, cs = np.zeros((H,w,sd)), np.zeros((H,w))
        As = np.random.randint(3,size=(H,w))
        Ss, Cs = np.zeros((H,w,sd)), np.zeros((H,w))
        s = env.reset()
        for h in range(H):
            Ss[h]=s
            c, s = env.step(As[h])
            Cs[h]=c
        succ=np.flatnonzero(1-c)[:suc]
        fail=np.flatnonzero(c)[:fa]
        idx = np.append(succ,fail)
        sl = np.s_[:,-suc-fa:tot-suc-fa+idx.size]
        data[sl]=Ss[:,idx],As[:,idx],Cs[:,idx]
        # Ss[sl],As[sl],Cs[sl]=ss[:,idx],as_[:,idx],cs[:,idx]
        suc-=succ.size ; fa-=fail.size
    return data[:,np.argsort(data[-1,:,-1])]
#    return list(zip(Ss,As,Cs))

def eval(env, theta):
    w=env.width ; s=env.reset(1)
    cost=0
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

def exper(envtype, envparams, datasiz, succsiz):
    cs=np.zeros((len(succsiz),len(datasiz),2))
    env=envtype(*envparams)
    A = 2 if envtype is CartPole else 3
    for i in range(len(succsiz)):
        data=getdata(env, sorted(datasiz)[-1], s)
        # todo sort data here
        for j in range(len(datasiz)):
            thetalog=fqi(data[:d],A,'log')
            thetasq=fqi(data[:d],A,'sq')
            cs[i,j]=eval(env,thetalog),eval(env,thetasq)
