import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym.envs
from itertools import count
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
random.seed(100)
env.seed(100)

N_ACTIONS = env.action_space.shape[0]
R_ACTIONS = env.action_space.high[0] # range
N_STATES = env.observation_space.shape[0]
print(N_ACTIONS, R_ACTIONS, N_STATES)

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 1e-3
ou_theta, ou_sigma, ou_mu = 0.15, 0.2, 0.0
BATCH_SIZE = 64
MEMORY_CAPACITY = 500000


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(N_STATES, 400)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(400, 300)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(300, N_ACTIONS)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.before_action = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(N_STATES, 400)),
            nn.ReLU()
        )
        self.after_action = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(400 + N_ACTIONS, 300)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(300, 1))
        )

    def forward(self, x, action):
        x = self.before_action(x)
        x = torch.cat([x, action], dim=1)
        x = self.after_action(x)
        return x


class DDPG(nn.Module):
    def __init__(self):
        super(DDPG, self).__init__()
        self.actor = Actor()
        self.actor_target = Actor()
        self.critic = Critic()
        self.critic_target = Critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.ou = OrnsteinUhlenbeckActionNoise(theta=ou_theta, sigma=ou_sigma, mu=ou_mu, action_dim=N_ACTIONS) # explortion


    def get_action(self, state):
        state = torch.from_numpy(state).float()
        model_action = self.actor(state).detach().numpy() * R_ACTIONS
        action = model_action + self.ou.sample() * R_ACTIONS
        return action

    def update_target_model(self):  # soft_update
        target = self.actor_target
        source = self.actor
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        target = self.critic_target
        source = self.critic
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((deepcopy(state), action, reward, deepcopy(next_state), done))

    def _get_sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def train(self):
        minibatch = np.array(self._get_sample(BATCH_SIZE)).transpose()

        states = np.vstack(minibatch[0])
        actions = np.vstack(minibatch[1])
        rewards = np.vstack(minibatch[2])
        next_states = np.vstack(minibatch[3])
        dones = np.vstack(minibatch[4].astype(int))

        rewards = torch.Tensor(rewards)
        dones = torch.Tensor(dones)
        actions = torch.Tensor(actions)

        # critic update
        self.critic_optimizer.zero_grad()
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        next_actions = self.actor_target(next_states)

        pred = self.critic(states, actions)
        next_pred = self.critic_target(next_states, next_actions)

        target = rewards + (1 - dones) * GAMMA * next_pred
        critic_loss = F.mse_loss(pred, target)
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        pred_actions = self.actor(states)
        actor_loss = self.critic(states, pred_actions).mean()
        actor_loss = -actor_loss
        actor_loss.backward()
        self.actor_optimizer.step()


ddpg = DDPG()

def main():
    flag = False
    total_reward_list = []
    memory_counter = 0
    finish_reward = -130

    for i_episode in count(1):
        total_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, N_STATES])
        if (flag == True):
            break

        for t in range(700):
            memory_counter += 1
            action = ddpg.get_action(state)

            next_state, reward, done, info = env.step([action])
            next_state = np.reshape(next_state, [1, N_STATES])
            reward = float(reward[0, 0])
            ddpg.append_sample(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if memory_counter > BATCH_SIZE:
                ddpg.train()
                ddpg.update_target_model()

            if done:
                total_reward_list.append(total_reward)
                print(i_episode, "episode,  ", "moving average:", [int(i) for i in total_reward_list[-3:]], "=",
                      np.mean(total_reward_list[-3:], dtype=int))
                break

        if i_episode % 10 == 0:
            print(datetime.now())

        if np.mean(total_reward_list[-3:]) > finish_reward:
            env.render()
            print("save model start")
            torch.save(ddpg.state_dict(), './SaveModel/DDPG.pth')
            torch.save(ddpg.actor.state_dict(), './SaveModel/DDPG_actor.pth')
            torch.save(ddpg.critic.state_dict(), './SaveModel/DDPG_critic.pth')
            plt.plot(list(range(len(total_reward_list))), total_reward_list, c="b", lw=3, ls="--")
            plt.xlabel("Episode"), plt.ylabel("Reward"), plt.title("Reward Graph")
            fig = plt.gcf()
            fig.savefig('./SaveModel/DDPG_Reward_Graph.png')
            plt.close()

            if np.mean(total_reward_list[-20:]) > (finish_reward+40):
                print("finish")

                torch.save(ddpg.state_dict(), './SaveModel/DDPG_save.pth')
                torch.save(ddpg.actor.state_dict(), './SaveModel/DDPG_actor_save.pth')
                torch.save(ddpg.critic.state_dict(), './SaveModel/DDPG_critic_save.pth')
                flag = True
                break

if __name__ == '__main__':
    main()