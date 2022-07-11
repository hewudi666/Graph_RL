import numpy as np
import matplotlib.pyplot as plt

algorithm = ['No resolution', 'MADDPG', 'PPO', 'GRL']
agent_num = [8, 15, 20, 25, 30, 35, 40, 50]
collision_num = [
        [2.2,8.43,14.56,22.41,34.07,42.1,57.77,89.83],
        [0.24,3.47,9.82,12.67,20.42,29.16,45.19,65.0],
        [0.61,3.9,8.27,14.85,18.85,28.06,41.91,68.81],
        [0.1,2.15,6.99,8.83,15.02,19.97,26.37,44.84]
    ]

# 刻度在内，设置刻度字体大小
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
color = ['dodgerblue','sandybrown','palegreen','violet']
marker=['>','^','s','*']

for i in range(len(algorithm)):
    plt.plot(agent_num, collision_num[i], label=algorithm[i], linewidth=2, linestyle='-', marker=marker[i], c=color[i], markersize=8)

# x轴,y轴设置
names = ['8', '15', '20', '25','30','35','40','50']
plt.tick_params(labelsize=15, width=1.5, gridOn=True)
plt.xticks(agent_num, names)
plt.xlabel('Number of aircraft in the airspace', fontsize=15)
plt.ylabel('CN',fontsize=15)
# 边框线宽
ax = plt.gca()
bwith = 1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(bottom='on', left='on')
plt.legend(prop={"family" : "Times New Roman",'size': 15})

plt.savefig("collision_number.png", dpi=600)
plt.show()