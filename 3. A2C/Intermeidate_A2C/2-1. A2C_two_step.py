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

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=100, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=650.0,
)

env = gym.make('CartPole-v5')
env.seed(args.seed)
torch.manual_seed(args.seed)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
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
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.log_p.append(m.log_prob(action))
    model.values.append(state_value)

    return action.item()


def finish_episode(rewards,next_state, done):
    policy_losses = []
    value_losses = []
    next_state = torch.from_numpy(next_state).float()
    next_value = model.critic(next_state)

    rewards = torch.tensor(rewards).float()
    value = model.values[0]

    if done:
        advantage = rewards[1] - value
        target = rewards[1]
    else:
        advantage = (rewards[0] + args.gamma * (rewards[1] + args.gamma * next_value)) - value
        target = rewards[0] + args.gamma * (rewards[1] + args.gamma * next_value)

    advantage = torch.tensor(advantage, requires_grad=True)
    target = torch.tensor(target, requires_grad=True)

    policy_losses.append(-(model.log_p[0] * advantage))
    #policy_losses.append(-(model.log_p[0] * model.log_p[1] * advantage))
    value_losses.append((target - value) ** 2)

    model.actor_optim.zero_grad()
    loss1 = torch.stack(policy_losses).sum()
    loss1.backward(retain_graph=True)
    model.actor_optim.step()

    model.critic_optim.zero_grad()
    loss2 = torch.stack(value_losses).sum()
    loss2.backward(retain_graph=True)
    model.critic_optim.step()


def main():
    total_reward_list = []
    for i_episode in count(1):
        rewards = []
        total_reward = 0
        state = env.reset()

        for t in range(1,800):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward

            if t % 2 == 0:
                finish_episode(rewards,state, done)
                del rewards[:]
                del model.log_p[:]
                del model.values[:]

            if done:
                total_reward_list.append(total_reward)
                del model.log_p[:]
                del model.values[:]
                break

        # evaluation

        if i_episode % 5 == 0:
            print(i_episode,"번째:",total_reward_list[-5:], ", mean:", np.mean(total_reward_list[-5:], dtype=int))

        if np.mean(total_reward_list[-5:], dtype=float) > env.spec.reward_threshold: # 695, line 31
            torch.save(model.state_dict(),'./SaveModel/2.actor_critic_two_step.pth')

            plt.plot(list(range(len(total_reward_list))), total_reward_list, c="b", lw=3, ls="--")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Reward Graph")
            fig = plt.gcf()
            plt.show()
            fig.savefig('./SaveModel/2.two_step_reward_graph.png')
            plt.close()
            break


if __name__ == '__main__':
    main()