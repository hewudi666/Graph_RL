import numpy as np
import matplotlib.pyplot as plt

dir = "C:\\Users\\lenovo\\PycharmProjects\\project\\GRL\\model\\"

maddpg_dir = dir + "cr_maddpg\\" + "30_train_returns_test_0330.npy"
ppo_dir = dir + "cr_ppo\\" + "30_train_returns_test1.npy"
grl_dir = dir + "cr_grl\\30_agent\\" + "30_train_returns_test_no_cm.npy"
# grl_dir = dir + "cr_grl\\30_agent\\" + "30_train_returns_dynamic.npy"


maddpg_c_dir = dir + "cr_maddpg\\" + "30_train_conflict_test.npy"
ppo_c_dir = dir + "cr_ppo\\" + "30_train_conflict_test1.npy"
grl_c_dir = dir + "cr_grl\\30_agent\\" + "30_conflict_num_test_no_cm.npy"

maddpg = np.load(maddpg_dir).tolist()
ppo = np.load(ppo_dir).tolist()
grl = np.load(grl_dir).tolist()

maddpg_c = np.load(maddpg_c_dir).tolist()
ppo_c = np.load(ppo_c_dir).tolist()
grl_c = (np.load(grl_c_dir) / 10).tolist()

L = len(maddpg)
x = (10 * np.array(range(L))).tolist()

# 画图设置
plt.figure('Coverage of different algorithms')

# 刻度在内，设置刻度字体大小
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
color = ['peru','royalblue','m']
marker=['>','*','s','^']

plt.plot(x, maddpg,color=color[0], label='MADDPG', linewidth=1.5, linestyle='-')
plt.plot(x, ppo[: L],color=color[1], label='PPO', linewidth=1.5, linestyle='-')
plt.plot(x, grl, color=color[2], label='GRL', linewidth=1.5, linestyle='-')
plt.tick_params(labelsize=15, width=1.5)
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Mean Reward', fontsize=18)
ax = plt.gca()
bwith = 1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.tick_params(bottom='on', left='on')
plt.legend(['MADDPG', 'PPO', 'GRL'], loc='lower right', prop={"family": "Times New Roman", 'size':15})

plt.savefig("reward curve.png", dpi=600)
plt.show()

# plt.figure()
# plt.plot(x, ppo_c[: L], 'b', x, grl_c, 'r', x, maddpg_c, 'g')
# plt.xlabel('episode')
# plt.ylabel('ave conflict')
# plt.legend(['ppo', 'grl', 'maddpg'])
# plt.show()