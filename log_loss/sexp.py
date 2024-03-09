import numpy as np
from sfqi import fqi

# tot - total trajectories ; suc - number of successes
def getdata(env, tot, suc=1):
    H, w, sd = env.H, env.width, env.getstate().shape[1]
    data=np.zeros((H,tot,sd+2))# stores state,action,cost
    fa=tot-suc
    while suc or fa:
        # sac=np.zeros((H,w,sd+2))
        # sac[:,:,-2] = np.random.randint(3,size=(H,w))
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
        suc-=succ.size ; fa-=fail.size
    return data[:,np.argsort(data[-1,:,-1])]

def eval(env, theta):
    w=env.width ; s=env.reset(1)
    cost=0
    for h in range(env.H):
        c, s = env.step(np.argmin(theta[h]@s.T))
        cost+=c
    env.reset(w)
    return cost

# def exper(envtype, envparams, dataparams):
#     A = 2 if envtype is CartPole else 3
#     env=envtype(*envparams)
#     data=getdata(env, *dataparams)
#     thetalog = fqi(data,A,'log')
#     thetasq = fqi(data,A,'sq')
#     return eval(env,thetalog), eval(env,thetasq)

def exper(envtype, envparams, datasiz, succsiz):
    env=envtype(*envparams)
    cs=np.zeros((len(succsiz),len(datasiz),2))
    for i in range(len(succsiz)):
        data=getdata(env,sorted(datasiz)[-1],succsiz[i])
        for j in range(len(datasiz)):
            sac=zip(data[:,:datasiz[j],:-2],
                    data[:,:datasiz[j],-2],
                    data[:,:datasiz[j],-1])
            thetalog=fqi(sac,env.A,'log')
            thetasq=fqi(sac,env.A,'sq')
            cs[i,j]=eval(env,thetalog),eval(env,thetasq)
    return cs
