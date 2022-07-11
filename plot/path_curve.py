import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

data_x = np.load('E:\\buaa\\冲突解脱\\实验相关\\经典场景测试\\数据收集\\graph_save_16_圆形对飞\\data\\1073\\data_x.npy')
data_y = np.load('E:\\buaa\\冲突解脱\\实验相关\\经典场景测试\\数据收集\\graph_save_16_圆形对飞\\data\\1073\\data_y.npy')

# 整理agent位置信息
agent_pos = []
agent_num = len(data_x[0])
T = len(data_x)
agent_radius = 5
for i in range(T):
    pos = []
    for j in range(agent_num):
        x = data_x[i][j]
        y = data_y[i][j]
        pos.append((x, y))
    agent_pos.append(pos)

# 刻度在内，设置刻度字体大小
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
cmap = plt.cm.get_cmap('hsv', 10)
goal_color = 'blue'
start_color = 'black'

fig, ax = plt.subplots(figsize=(6.5,6))
boundary = 160
# plt.axis('equal')
ax.tick_params(labelsize=16)
ax.set_xlim(-(boundary + 10), boundary + 10)
ax.set_ylim(-(boundary + 10), boundary + 10)
ax.set_xlabel('x(nautical miles)', fontsize=16)
ax.set_ylabel('y(nautical miles)', fontsize=16)

goal_x = [p[0] for p in agent_pos[-1]]
goal_y = [p[1] for p in agent_pos[-1]]
start_x = [p[0] for p in agent_pos[0]]
start_y = [p[1] for p in agent_pos[0]]
goal = mlines.Line2D(goal_x, goal_y, color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
start = mlines.Line2D(start_x, start_y, color=start_color, marker='o', linestyle='None', markersize=6, label='Start')
collision_num = plt.text(-55, boundary + 20, 'Collision Num: 0', color='red', fontsize=16)
ax.add_artist(collision_num)
ax.add_artist(goal)
ax.add_artist(start)

for k in range(T):
    if k % 6 == 0 or k == T - 1:
        agents = [plt.Circle(agent_pos[k][i], agent_radius, fill=False, color=cmap(i % 10))
                  for i in range(agent_num)]

        for i in range(agent_num):
            ax.add_artist(agents[i])

    if k != 0 and k % 5 == 0:
        nav_dirs = [plt.Line2D(
            (agent_pos[k - 4][i][0], agent_pos[k][i][0]),
            (agent_pos[k - 4][i][1], agent_pos[k][i][1]),
            color=cmap(i))
            for i in range(agent_num)]

        for nav_dir in nav_dirs:
            ax.add_artist(nav_dir)

plt.tight_layout()
save_path = "E:\\buaa\\冲突解脱\\实验相关\\经典场景测试\\"
plt.savefig(save_path + "16个智能体圆对飞场景.png", dpi=600)
plt.show()






