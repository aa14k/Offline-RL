import numpy as np
import matplotlib.pyplot as plt
import datetime

c_log = np.load('results/3c_log_2024-01-27 22:20:30.304386.npy')
c_sq = np.load('results/3c_sq_2024-01-27 22:20:30.304386.npy')


c_log =np.sum(c_log,axis=1) - 0.5
c_sq =np.sum(c_sq,axis=1) 
#print(c_log)


data = np.array([30000, 27000, 24000, 21000, 18000, 15000, 12000, 9000, 6000, 3000, 1000])

runs = 90 

current_time = datetime.datetime.now()
np.save('results/c_log_'+ str(current_time), c_log)
np.save('results/c_sq_'+ str(current_time), c_sq)

plt.figure(figsize=(7,4))
plt.scatter(data, c_log / runs, label = 'log')
plt.scatter(data, c_sq / runs, label='sq')
plt.xlabel('Number of trajectories')
plt.ylabel('$V(\pi_{FQI})$')
plt.legend()
#plt.title('Performance of FQI vs Size of Dataset Averaged over ' + str(runs) + ' runs.')
plt.savefig('results/mc_plot_' + str(current_time) + '.pdf')
