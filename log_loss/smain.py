import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sexp import exper
from senv import MountainCar, CartPole, Acrobot
import datetime

# env params
H = 800
order = 2
simwidth=10000
feat='f' # f or p
envparams=(H,order,simwidth,feat)
# exp params
envtype=MountainCar
#datasiz=[1000, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
datasiz = [1000,3000,6000]#[15000,12000,9000,6000,3000,1000]
#succsiz=[1,5,30]
succsiz=[1]
trials=90

if __name__=='__main__':
    c=np.array(Parallel(n_jobs=-3)(
        delayed(exper)(envtype, envparams,
                       datasiz, succsiz)
        for _ in range(trials)))
    clog, csq = map(lambda x:np.squeeze(x,-1), np.split(c,2,-1))
    time=str(datetime.datetime.now())
    np.save('sresults/clog-'+time, clog)
    np.save('sresults/csq-'+time, csq)
    for i in range(len(succsiz)):
        plt.plot(datasiz,clog.sum(0)[i]/trials,label='log')
        plt.plot(datasiz,csq.sum(0)[i]/trials,label='sq')
        plt.xlabel('Number of trajectories')
        plt.ylabel('$V(\pi_{FQI})$')
        plt.legend()
        plt.title('Performance vs Size of Dataset over '\
                  f'{trials} trials with {succsiz[i]} successful traj.')
        plt.savefig(f'sresults/mcplot-{succsiz[i]}-{time}.pdf')
        plt.close()
