import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dgn.DGN import DGN
from dgn.dgn_r.buffer import ReplayBuffer
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import time

class CosineSimilarity(nn.Module):

    def forward(self, tensor_1, tensor_2):
        norm_tensor_1 = tensor_1.norm(dim=-1, keepdim=True)
        norm_tensor_2 = tensor_2.norm(dim=-1, keepdim=True)
        norm_tensor_1 = norm_tensor_1.numpy()
        norm_tensor_2 = norm_tensor_2.numpy()

        for i, vec2 in enumerate(norm_tensor_1[0]):
            for j, scalar in enumerate(vec2):
                if scalar == 0:
                    norm_tensor_1[0][i][j] = 1
        for i, vec2 in enumerate(norm_tensor_2[0]):
            for j, scalar in enumerate(vec2):
                if scalar == 0:
                    norm_tensor_2[0][i][j] = 1
        norm_tensor_1 = torch.tensor(norm_tensor_1)
        norm_tensor_2 = torch.tensor(norm_tensor_2)
        normalized_tensor_1 = tensor_1 / norm_tensor_1
        normalized_tensor_2 = tensor_2 / norm_tensor_2
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

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
        self.n_action = 5
        self.hidden_dim = 128
        self.buffer = ReplayBuffer(args.buffer_size, 9, self.n_action, self.agent_num)
        self.lr = 1e-4
        self.batch_size = args.batch_size
        self.train_epoch = 25
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
        self.model_name = '/8_agent/8_graph_rl_weight.pth'
        if os.path.exists(self.save_path + self.model_name):
            self.model.load_state_dict(torch.load(self.save_path + self.model_name))
            print("successfully load model: {}".format(self.model_name))

    def js_div(self, p_output, q_output, get_softmax=True):
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output, dim=-1)
            q_output = F.softmax(q_output, dim=-1)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

    def adj_window(self, adj_acc):
        T = 5
        adj = adj_acc[-1]
        if len(adj_acc) < T:
            return adj
        gamma = 0.8
        for t in range(1, T):
            adj += adj_acc[-(t + 1)] * (pow(gamma, t))
        return adj / T

    def run(self):
        tau = 0.98
        reward_total = []
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
            adj_ave = []
            obs, adj = self.env.reset()
            print("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    step += 1
                    action = []
                    obs1 = np.expand_dims(obs, 0)
                    adj1 = np.expand_dims(adj, 0)
                    adj_ave.append(adj1)
                    adj1 = self.adj_window(adj_ave)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]
                    # 待改
                    for i, agent in enumerate(self.agents):
                        if np.random.rand() < self.epsilon:
                            a = np.random.randint(self.n_action)
                        else:
                            a = q[i].argmax().item()
                        action.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(action)

                    self.buffer.add(obs, action, reward, next_obs, adj, next_adj, info['simulation_done'])
                    obs = next_obs
                    adj = next_adj

                else:
                    # print(" agent_terminated_times:", self.env.agent_times)
                    if self.env.simulation_done:
                        print("all agents done!")
                    break

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                if episode % (5 * self.args.evaluate_rate) == 0:
                    self.env.render(mode='traj')
                reward_total.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])
            self.env.conflict_num_episode = 0
            self.env.nmac_num_episode = 0

            if episode < start_episode:
                continue

            for epoch in range(self.train_epoch):

                Obs, Act, R, Next_Obs, Mat, Next_Mat, D = self.buffer.getBatch(self.batch_size)
                Obs = torch.Tensor(Obs).cuda()
                Mat = torch.Tensor(Mat).cuda()
                Next_Obs = torch.Tensor(Next_Obs).cuda()
                Next_Mat = torch.Tensor(Next_Mat).cuda()
                q_values = self.model(Obs, Mat)
                target_q_values = self.model_tar(Next_Obs, Next_Mat)
                target_q_values = target_q_values.max(dim=2)[0]
                target_q_values = np.array(target_q_values.cpu().data)
                expected_q = np.array(q_values.cpu().data)

                for j in range(self.batch_size):
                    for i in range(self.agent_num):
                        expected_q[j][i][Act[j][i]] = R[j][i] + (1 - D[j]) * self.gamma * target_q_values[j][i]


                loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for p, p_targ in zip(self.model.parameters(), self.model_tar.parameters()):
                        p_targ.data.mul_(tau)
                        p_targ.data.add_((1 - tau) * p.data)

            if episode != 0 and episode % 100 == 0:
                torch.save(self.model.state_dict(), rl_model_dir)
                print("torch save model for rl_weight")

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
            adj_ave = []
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    adj_ave.append(adj1)
                    adj1 = self.adj_window(adj_ave)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]  # shape (100, 5)
                    for i, agent in enumerate(self.agents):
                        # a = np.random.randint(self.n_action)
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

        return sum(returns) / self.args.evaluate_episodes, (
        self.env.collision_num / self.args.evaluate_episodes, self.env.exit_boundary_num / self.args.evaluate_episodes,
        self.env.success_num / self.args.evaluate_episodes, self.env.nmac_num / self.args.evaluate_episodes)


    def evaluate_model(self):
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
            adj_ave = []
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    actions = []
                    obs1 = np.expand_dims(obs, 0)
                    adj1 = np.expand_dims(adj, 0)
                    adj_ave.append(adj1)
                    adj1 = self.adj_window(adj_ave)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]
                    for i, agent in enumerate(self.agents):
                        a = q[i].argmax().item()
                        actions.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    rewards += sum(reward)
                    obs = next_obs
                    adj = next_adj
                else:
                    dev = self.env.route_deviation_rate()
                    if dev:
                        deviation.append(np.mean(dev))
                    break

            if episode > 0 and episode % 50 == 0:
                self.env.render(mode='traj')

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

