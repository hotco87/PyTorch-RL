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
        self.ou = OrnsteinUhlenbeckActionNoise(theta=ou_theta, sigma=ou_sigma, mu=ou_mu, action_dim=N_ACTIONS)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action = self.actor(state).detach().numpy() * R_ACTIONS
        return action

ddpg = DDPG()
ddpg.actor.load_state_dict(torch.load('./SaveModel/DDPG_actor.pth'))

def main():

    for i_episode in count(1):
        state = env.reset()
        print(state)
        state = np.reshape(state, [1, N_STATES])

        for t in range(20000):
            env.render()
            action = ddpg.get_action(state)
            next_state, reward, done, info = env.step([action])
            state = np.reshape(next_state, [1, N_STATES])

            if done:
                break

if __name__ == '__main__':
    main()