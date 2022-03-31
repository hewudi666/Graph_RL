import numpy as np
import matplotlib.pyplot as plt

dir = "C:\\Users\\lenovo\\PycharmProjects\\project\\GRL\\model\\"

maddpg_dir = dir + "cr_maddpg\\" + "30_train_returns_test.npy"
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
x = range(L)


plt.figure()
plt.plot(x, ppo[: L], 'g', x, grl, 'r', x, maddpg, 'b')
plt.xlabel('episode')
plt.ylabel('ave reward')
plt.legend(['ppo', 'grl', 'maddpg'])
plt.show()

# plt.figure()
# plt.plot(x, ppo_c[: L], 'b', x, grl_c, 'r', x, maddpg_c, 'g')
# plt.xlabel('episode')
# plt.ylabel('ave conflict')
# plt.legend(['ppo', 'grl', 'maddpg'])
# plt.show()