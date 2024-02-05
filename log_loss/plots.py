import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd 
import seaborn as sns


params = {
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'xtick.major.pad': 20,
    'ytick.labelsize': 20,
    'ytick.major.pad': 20,
    'legend.fontsize': 20,
    'legend.fontsize': 20,
}




runs = 90 

num_successes = [1,5,30]

for num_success in num_successes:

    if num_success == 1:
        c_log = np.array(np.load('results/c_log_2024-01-29 18:24:36.440977_H_800_1.npy') )
        c_sq = np.array(np.load('results/c_sq_2024-01-29 18:24:36.440977_H_800_1.npy') )

    if num_success == 5:
        c_log = np.array(np.load('results/c_log_2024-01-29 22:41:44.059894_H_800_5.npy') )
        c_sq = np.array(np.load('results/c_sq_2024-01-29 22:41:44.059894_H_800_5.npy') )
    if num_success == 30:
        c_log = np.array(np.load('results/c_log_2024-01-30 15:51:48.240481_H_800_30.npy') )
        c_sq = np.array(np.load('results/c_sq_2024-01-30 15:51:48.240481_H_800_30.npy') )

    for i in range(len(c_log)):
        for j in range(len(c_log[0])):
            c_log[i,j] = c_log[i,j] 
            c_sq[i,j] = c_sq[i,j] 

    #c_log =np.sum(c_log,axis=1) / runs 
    #c_sq =np.sum(c_sq,axis=1) / runs
    #print(c_log)


    datas = np.array([30000, 27000, 24000, 21000, 18000, 15000, 12000, 9000, 6000, 3000, 1000])
    n = len(datas)

    df = pd.DataFrame(columns=['Trajectories', 'Value', 'Loss'])

    for i in range(n):
        for j in range(len(c_log[0])):
            df = df.append({'Trajectories': datas[i], 'Value': c_log[i,j], 'Loss':'FQI-LOG'},ignore_index=True)
            df = df.append({'Trajectories': datas[i], 'Value': c_sq[i,j], 'Loss':'FQI-SQ'},ignore_index=True)





    current_time = datetime.datetime.now()


    plt.figure(figsize=(6,4))
    sns.set(font_scale=1.45)
    sns.set_style('whitegrid')
    #sns.pointplot(data=df, x='Trajectories', y='Value', hue='Loss', linestyles=['--','--'], errorbar='se')
    ax = sns.lineplot(data=df, x='Trajectories', y='Value', hue='Loss', style='Loss', markers=['o','o'])
    ax.legend().set_title('')
    plt.xlabel('Trajectories')
    if num_success != 1:
        ax.set(ylabel='')
    if num_success == 1:
        plt.ylabel('Value')
    ax.lines[0].set_linestyle('dotted')
    ax.lines[1].set_linestyle('dotted')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if num_success != 1:
        plt.legend([],[], frameon=False)
    plt.ylim(-0.05, 0.95)
    #plt.title('Performance of FQI vs Size of Dataset Averaged over ' + str(runs) + ' runs.')
    plt.tight_layout()
    plt.savefig('results/sns_plot_' + str(num_success) + '.pdf')# + str(current_time) + '.pdf')
