import numpy as np
from sfqi import fqi

# tt - total trajectories ; st - total successes
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

def eval(env, theta):
    w=env.width ; s=env.reset(1)
    cost=0
    for h in range(env.H):
        c, s = env.step(np.argmin(theta[h]@s.T))
        cost+=c
    env.reset(w)
    return cost.squeeze()

def exper(envtype, envparams, datasiz, succsiz):
    env=envtype(*envparams)
    cs=np.zeros((len(succsiz),len(datasiz),2))
    for i in range(len(succsiz)):
        data=getdata(env,sorted(datasiz)[-1],succsiz[i])
        for j in range(len(datasiz)):
            sac=list(zip(data[:,:datasiz[j],:-2],
                         data[:,:datasiz[j],-2],
                         data[:,:datasiz[j],-1]))
            thetalog=fqi(sac,env.A,'log')
            thetasq=fqi(sac,env.A,'sq')
            cs[i,j]=eval(env,thetalog),eval(env,thetasq)
    return cs
