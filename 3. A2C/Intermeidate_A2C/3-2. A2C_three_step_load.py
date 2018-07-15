# Based on: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gym.envs.registration import register
import matplotlib.pyplot as plt


register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=650.0,
)

env = gym.make('CartPole-v5')

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=-1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        return state_values


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=5e-3)
        self.log_p = []
        self.values = []
        self.value = 0

    def forward(self, x):
        return self.actor(x), self.critic(x)


model = A2C()
model.load_state_dict(torch.load('./SaveModel/2.actor_critic_three_step.pth'))

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.log_p = m.log_prob(action)
    model.cur_value = state_value
    return action.item()


def main():
    state = env.reset()
    for t in range(10000):
        env.render()
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        print(t)
        if done:
            print("done")
            env.close()
            break
    env.close()

if __name__ == '__main__':
    main()
