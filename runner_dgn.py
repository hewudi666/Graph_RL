from dgn.buffer import ReplayBuffer
import torch
import torch.optim as optim
from dgn.DGN import DGN
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import time


class Runner_DGN:
    def __init__(self, args, env):
        self.args = args
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)
        USE_CUDA = torch.cuda.is_available()
        self.env = env
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.num_episode = args.num_episodes
        self.max_step = args.max_episode_len
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.buffer = ReplayBuffer(args.buffer_size)
        self.n_action = 5
        self.hidden_dim = 128
        self.lr = 1e-4
        self.batch_size = args.batch_size
        self.train_epoch = 10
        self.gamma = args.gamma
        self.observation_space = self.env.observation_space
        self.model = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action)
        self.model_tar = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action)
        self.model = self.model.cuda()
        self.model_tar = self.model_tar.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # self.output_file = self.save_path + '/15_agent/video/test.gif'
        self.model_name = '/30_agent/30_graph_rl_weight.pth'
        if os.path.exists(self.save_path + self.model_name):
            self.model.load_state_dict(torch.load(self.save_path + self.model_name))
            print("successfully load model: {}".format(self.model_name))


    def run(self):
        Obs = np.ones((self.batch_size, self.agent_num, self.observation_space))
        Next_Obs = np.ones((self.batch_size, self.agent_num, self.observation_space))
        matrix = np.ones((self.batch_size, self.agent_num, self.agent_num))
        next_matrix = np.ones((self.batch_size, self.agent_num, self.agent_num))

        reward_total = []
        reward_total_t = []
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []
        start_episode = 40
        start = time.time()
        episode = -1
        rl_model_dir = self.save_path + self.model_name
        while episode < self.num_episode:
            if episode > start_episode:
                self.epsilon = max(0.05, self.epsilon - self.epsilon_decay)

            episode += 1
            step = 0
            obs, adj = self.env.reset()

            print("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    # print(" {} episode {} step ".format(i_episode, steps))
                    step += 1
                    action = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]  # shape (agent_num, action_num)
                    # 待改
                    for i, agent in enumerate(self.agents):
                        if np.random.rand() < self.epsilon:
                            a = np.random.randint(self.n_action)
                        else:
                            a = q[i].argmax().item()
                        action.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(action)

                    self.buffer.add(obs, action, reward, next_obs, adj, next_adj, done_signals)
                    obs = next_obs
                    adj = next_adj

                else:
                    # print(" agent_terminated_times:", self.env.agent_times)
                    if self.env.simulation_done:
                        print("all agents done!")
                    break

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew_t, rew, info = self.evaluate()
                if episode % (5 * self.args.evaluate_rate) == 0:
                    self.env.render(mode='traj')
                reward_total.append(rew)
                reward_total_t.append(rew_t)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])
            self.env.conflict_num_episode = 0
            self.env.nmac_num_episode = 0

            if episode < start_episode:
                continue

            for epoch in range(self.train_epoch):
                batch = self.buffer.getBatch(self.batch_size)
                for j in range(self.batch_size):
                    sample = batch[j]
                    Obs[j] = sample[0]
                    Next_Obs[j] = sample[3]
                    matrix[j] = sample[4]
                    next_matrix[j] = sample[5]

                q_values = self.model(torch.Tensor(Obs).cuda(), torch.Tensor(matrix).cuda())  # shape (128, 6, 3)
                target_q_values = self.model_tar(torch.Tensor(Next_Obs).cuda(), torch.Tensor(next_adj).cuda()).max(dim=2)[0]  # shape  (128, 6)
                target_q_values = np.array(target_q_values.cpu().data)  # shape  (128, 6)
                expected_q = np.array(q_values.cpu().data)  # (batch_size, agent_num, action_num)

                for j in range(self.batch_size):
                    sample = batch[j]
                    for i in range(self.agent_num):
                        # sample[1]: action selection list ; sample[2]: reward size-agent_num ; sample[6]: terminated
                        # expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * self.gamma * target_q_values[j][i]
                        if sample[6][i] != 1:
                            expected_q[j][i][sample[1][i]] = sample[2][i] + self.gamma * target_q_values[j][i]
                        else:
                            expected_q[j][i][sample[1][i]] = sample[2][i]

                loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if episode % 5 == 0:
                self.model_tar.load_state_dict(self.model.state_dict())

            if episode != 0 and episode % 200 == 0:
                torch.save(self.model.state_dict(), rl_model_dir)
                print("torch save model for rl_weight")

        end = time.time()
        print("花费时间:", end - start)
        plt.figure()
        plt.plot(range(1, len(reward_total)), reward_total[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/30_agent/30_train_returns_test_no_cm1.png', format='png')
        np.save(self.save_path + '/30_agent/30_train_returns_test_no_cm1', np.array(reward_total))
        np.save(self.save_path + '/30_agent/30_train_returns_total_test_no_cm1', np.array(reward_total_t))

        fig, a = plt.subplots(2, 2)
        plt.title('GRL_train')
        x = range(len(conflict_total))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        a[1][1].plot(x, nmac_total)
        a[1][1].set_title('nmac_num')
        plt.savefig(self.save_path + '/30_agent/train_metric_test_no_cm1.png', format='png')
        np.save(self.save_path + '/30_agent/30_conflict_num_test_no_cm1', np.array(conflict_total))
        plt.show()

    def evaluate(self):
        print("now is evaluate!")
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]  # shape (100, 5)
                    for i, agent in enumerate(self.agents):
                        a = q[i].argmax().item()
                        actions.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    rewards += sum(reward)
                    obs = next_obs
                    adj = next_adj
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)

        print("平均conflict num :", self.env.collision_num / self.args.evaluate_episodes)
        print("平均nmac num :", self.env.nmac_num / self.args.evaluate_episodes)
        print("平均exit boundary num：", self.env.exit_boundary_num / self.args.evaluate_episodes)
        print("平均success num：", self.env.success_num / self.args.evaluate_episodes)
        print("路径平均偏差率：", np.mean(deviation))

        return returns, sum(returns) / self.args.evaluate_episodes, (self.env.collision_num / self.args.evaluate_episodes, self.env.exit_boundary_num / self.args.evaluate_episodes, self.env.success_num / self.args.evaluate_episodes, self.env.nmac_num / self.args.evaluate_episodes)

    def evaluate_model(self):
        """
        对现有最新模型进行评估
        :return:
        """
        print("now evaluate the model")
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []
        deviation = []
        self.env.collision_num = 0
        self.env.nmac_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        returns = []
        eval_episode = 100
        for episode in range(eval_episode):
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]  # shape (100, 5)
                    for i, agent in enumerate(self.agents):
                        a = q[i].argmax().item()
                        actions.append(a)
                        # print("agent {} action {}".format(i, a))

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    rewards += sum(reward)
                    obs = next_obs
                    adj = next_adj
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break
            # np.save(self.save_path + '/20_agent/actions/' + str(episode) + 'actions.npy',
            #         np.array(self.env.actions_total))

            if episode > 0 and episode % 50 == 0:
                # self.env.render(mode='video', output_file=self.output_file)
                self.env.render(mode='traj')
            # if episode > 0:
            #     self.env.render(mode='traj')

            # plt.figure()
            # plt.title('collision_value——time')
            # x = range(len(self.env.collision_value))
            # plt.plot(x, self.env.collision_value)
            # plt.xlabel('timestep')
            # plt.ylabel('collision_value')
            # plt.savefig(self.save_path + '/30_agent/collision_value/' + str(episode) + 'collision_value.png', format='png')
            # np.save(self.save_path + '/30_agent/collision_value/' + str(episode) + 'collision_value.npy', self.env.collision_value)
            # plt.close()

            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
            print("conflict num :", self.env.collision_num)
            print("nmac num：", self.env.nmac_num)
            print("exit boundary num：", self.env.exit_boundary_num)
            print("success num：", self.env.success_num)
            conflict_total.append(self.env.collision_num)
            nmac_total.append(self.env.nmac_num)
            collide_wall_total.append(self.env.exit_boundary_num)
            success_total.append(self.env.success_num)
            self.env.collision_num = 0
            self.env.nmac_num = 0
            self.env.exit_boundary_num = 0
            self.env.success_num = 0

        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        # plt.savefig(self.save_path + '/30_agent/eval_return_new.png', format='png')

        # conflict num process
        conflict_total_1 = []
        nmac_total_1 = []
        for i in range(len(conflict_total)):
            if success_total[i] + collide_wall_total[i] == self.agent_num:
                conflict_total_1.append(conflict_total[i])
                nmac_total_1.append(nmac_total[i])

        y = range(len(conflict_total))
        conflict_total = conflict_total_1
        nmac_total = nmac_total_1
        x = range(len(conflict_total))
        print("有效轮数：", len(x))
        fig, a = plt.subplots(2, 2)
        # 去除冲突数极大值
        conflict_total[conflict_total.index(max(conflict_total))] = 0
        conflict_total[conflict_total.index(max(conflict_total))] = 0
        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0) - 2
        print("平均冲突数", ave_conflict)
        print("平均NMAC数", ave_nmac)
        print("平均成功率", ave_success / self.agent_num)
        print("平均出界率", ave_exit / self.agent_num)
        print("0冲突占比：", zero_conflict / len(conflict_total))
        print("平均偏差率", np.mean(deviation))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(y, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(y, success_total, 'r')
        a[1][0].set_title('success_num')
        a[1][1].plot(x, nmac_total)
        a[1][1].set_title('nmac_num')
        # plt.savefig(self.save_path + '/50_agent/eval_metric2.png', format='png')

        plt.show()