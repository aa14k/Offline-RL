import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sexp import exper
from senv import MountainCar, CartPole, Acrobot
import datetime

# env params
H = 600
order = 2
simwidth=10000
feat='p' # f or p
envparams=(H,order,simwidth,feat)
# exp params
envtype=MountainCar
datasiz=[30000, 25000, 20000, 15000, 10000, 5000, 1000]
succsiz=1
runs=56

if __name__=='__main__':
    cs=np.zeros((len(datasiz),2,runs))
    for i in tqdm(range(len(datasiz))):
        tic=timer()
        c=Parallel(n_jobs=-3)(delayed(exper)(MountainCar, envparams,
                                             (datasiz[i], succsiz))
                              for j in tqdm(range(runs)))
        toc=timer()
        print(f'{np.sum(c,axis=0)}\nTime: {toc-tic}')
        cs[i]=np.array(c).T
        c.append(c)

    clog, csq=np.split(cs,2,1)
    time=str(datetime.datetime.now())
    np.save('sresults/c_log_'+time, clog)
    np.save('sresults/c_sq_'+time, csq)
    
    plt.plot(data,clog.sum(1)/runs,label='log')
    plt.plot(data,csq.sum(1)/runs,label='sq')
    plt.xlabel('Number of trajectories')
    plt.ylabel('$V(\pi_{FQI})$')
    plt.legend()
    plt.title(f'Performance of FQI vs Size of Dataset over {runs} runs with {succsiz} successful trajs.')
    plt.savefig(f'sresults/mc_plot_{time}.pdf')
    plt.legend()
