import numpy as np
import matplotlib.pyplot as plt

algorithm = ['No resolution', 'MADDPG', 'PPO', 'GRL']
agent_num = [8, 15, 20, 25, 30, 35, 40, 50]
path_deviation_rate = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0.26,1.10,1.39,2.52,2.16,3.24,3.68,5.14],
        [0.27,1.21,1.50,1.93,3.04,3.55,2.80,5.14],
        [0.12,0.91,1.18,1.73,0.70,2.38,3.59,4.68]
    ]

# 刻度在内，设置刻度字体大小
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
color = ['dodgerblue','sandybrown','palegreen','violet']
marker=['>','^','s','*']

for i in range(len(algorithm)):
    plt.plot(agent_num, path_deviation_rate[i], label=algorithm[i], linewidth=2, linestyle='-', marker=marker[i], c=color[i], markersize=8)

# x轴,y轴设置
names = ['8', '15', '20', '25','30','35','40','50']
plt.tick_params(labelsize=15, width=1.5, gridOn=True)
plt.xticks(agent_num, names)
plt.xlabel('Number of aircraft in the airspace', fontsize=15)
plt.ylabel('PD(%)',fontsize=15)
# 边框线宽
ax = plt.gca()
bwith = 1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(bottom='on', left='on')
plt.legend(prop={"family" : "Times New Roman", 'size':15})

plt.savefig("path_deviation.png", dpi=600)
plt.show()