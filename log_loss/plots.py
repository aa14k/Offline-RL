import numpy as np
import matplotlib.pyplot as plt
import datetime

c_log = np.array([0, 0, 1, 1, 2, 0, 1, 2, 0, 4, 11])
c_sq = np.array([16, 21, 13, 12, 7, 17, 31, 52, 63, 74, 77])

data = np.array([30000, 27000, 24000, 21000, 18000, 15000, 12000, 9000, 6000, 3000, 1000])

runs = 90 

current_time = datetime.datetime.now()
np.save('results/c_log_'+ str(current_time), c_log)
np.save('results/c_sq_'+ str(current_time), c_sq)

plt.figure(figsize=(7,4))
plt.bar(data, c_log / runs, label = 'log')
plt.bar(data, c_sq / runs, label='sq')
plt.xlabel('Number of trajectories')
plt.ylabel('$V(\pi_{FQI})$')
plt.legend()
#plt.title('Performance of FQI vs Size of Dataset Averaged over ' + str(runs) + ' runs.')
plt.savefig('results/mc_plot_' + str(current_time) + '.pdf')
