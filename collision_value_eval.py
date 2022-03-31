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
plt.figure()
plt.subplot(311)
plt.plot(x, maddpg, 'blue')
plt.title('maddpg')
plt.ylim([0, 3])
plt.xlabel('timestep')
plt.ylabel('collision_value')
plt.subplot(312)
plt.plot(x, ppo, 'green')
plt.title('ppo')
plt.ylim([0, 3])
plt.xlabel('timestep')
plt.ylabel('collision_value')
plt.subplot(313)
plt.plot(x, grl, 'red')
plt.title('grl')
plt.ylim([0, 3])
plt.xlabel('timestep')
plt.ylabel('collision_value')

# plt.show()
