import numpy as np
import matplotlib.pyplot as plt

dir = "C:\\Users\\lenovo\\PycharmProjects\\project\\GRL\\model\\"

maddpg_dir = dir + "cr_maddpg\\collision_value\\30_agent\\" + "20collision_value.npy"
ppo_dir = dir + "cr_ppo\\collision_value\\30_agent\\" + "58collision_value.npy"
grl_dir = dir + "cr_grl\\30_agent\\collision_value\\" + "37collision_value.npy"
# grl  37, 46, 78

maddpg = np.load(maddpg_dir).tolist()
ppo = np.load(ppo_dir).tolist()
grl = np.load(grl_dir).tolist()
max_len = 200
for i in range(200 - len(maddpg)):
    maddpg.append(0)
for i in range(200 - len(ppo)):
    ppo.append(0)
for i in range(200 - len(grl)):
    grl.append(0)


print("maddpg峰值：", max(maddpg))
print("ppo峰值：", max(ppo))
print("grl峰值：", max(grl))

x = range(len(maddpg))

plt.figure(figsize=[8.0, 6.0])
# plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tick_params(labelsize=15, width=1, gridOn=True)
plt.xticks(size=15)
plt.tick_params(bottom='on', left='on')
color = ['lightsalmon','lightgreen','lightskyblue']
marker=['^','s','*']
yticks = [0,0.5,1.0,1.5,2.0,2.5,3.0]

plt.subplot(311)
plt.plot(x, maddpg, label='MADDPG',linewidth=2, linestyle='-', c=color[0])
plt.grid(True)
plt.title('MADDPG', fontsize=15)
plt.ylim([0, 3])
plt.xlabel('Timestep', fontsize=15)
plt.ylabel('CV', fontsize=15)
plt.yticks(yticks)

plt.subplot(312)
plt.plot(x, ppo, label='PPO',linewidth=2, linestyle='-', c=color[1])
plt.grid(True)
plt.title('PPO', fontsize=15)
plt.ylim([0, 3])
plt.xlabel('Timestep', fontsize=15)
plt.ylabel('CV', fontsize=15)
plt.yticks(yticks)

plt.subplot(313)
plt.plot(x, grl, label='GRL',linewidth=2, linestyle='-', c=color[2])
plt.grid(True)
plt.title('GRL', fontsize=15)
plt.ylim([0, 3])
plt.xlabel('Timestep', fontsize=15)
plt.ylabel('CV', fontsize=15)
plt.yticks(yticks)

plt.tight_layout()
plt.savefig("collision_value.png",dpi=600)
plt.show()
