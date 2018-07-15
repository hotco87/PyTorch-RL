# -*- coding: utf-8 -*-
# Original Code : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from gym.envs.registration import register

# https://github.com/openai/gym/blob/master/gym/envs/__init__.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=695.0,
)

env = gym.make('CartPole-v5').unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
train_step_counter = 0  # for target updating
memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)

        #return self.head(actions_value.view(actions_value.size(0), -1))
        return actions_value

env.reset()

######################################################################
# Training
# Hyperparameters and utilities


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    state = torch.from_numpy(state).float().unsqueeze(0)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

episode_durations = []

# Training loop
# Finally, the code for training our model.

def train(self):
    # target parameter update
    if self.train_step_counter % TARGET_REPLACE_ITER == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict())
    self.train_step_counter += 1

    # sample batch transitions
    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = self.memory[sample_index, :]
    b_s = torch.FloatTensor(b_memory[:, :N_STATES])
    b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
    b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
    b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

    # q_eval w.r.t the action in experience
    q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
    q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
    q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
    loss = self.loss_func(q_eval, q_target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

def replay_memory(s, a, r, s_, memory_counter):
    transition = np.hstack((s, [a, r], s_)) # 4, 1, 1, 4 = size(10)
    # replace the old memory with new memory
    index = memory_counter % MEMORY_CAPACITY
    memory[index, :] = transition
    memory_counter += 1


# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation).
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.uint8)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                                 if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()

#num_episodes = 500

dqn = DQN()
  # for storing memory

def main():
    memory_counter = 0
    reward_threshold = 200
    total_reward_list = []
    for i_episode in count(1):

        # Initialize the environment and state
        state = env.reset()
        total_reward = 0
        # last_screen = get_screen()
        # current_screen = get_screen()
        # state = current_screen - last_screen
        total_reward = 0
        for t in range(700):
            # Select and perform an action
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward

            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 # 카트가 중간에 있을 수록 좋음.(카트와 중심 사이의 거리를 작게 하면 보상이 커짐.)
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # 막대기의 각도가 90도에 가까울 수록 보상이 커짐.
            reward = r1 + r2
            reward = torch.tensor([reward], device=device)
            # Observe new state
            #last_screen = current_screen
            #current_screen = get_screen()
            # if not done:
            #     next_state = current_screen - last_screen
            # else:
            #     next_state = None

            # Store the transition in memory
            #memory.push(state, action, next_state, reward)
            replay_memory(state, action, reward, next_state,memory_counter)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if memory_counter > MEMORY_CAPACITY and t % 4 == 0:     # 데이터가 모였을 때
            #if t>2000 and t%4==0:
                train()

            if done:
                episode_durations.append(t + 1)
                total_reward_list.append(total_reward)
                #print(total_reward)
                break
        # Update the target network
        if i_episode % 10 == 0: # 에피소드 10번 마다 보여주기
            print(i_episode,"번째:",total_reward_list[-5:], ", mean:", np.mean(total_reward_list[-5:], dtype=int))

        if np.mean(total_reward_list[-10:], dtype=int) > reward_threshold: # 이동평균, 695, line 31
            torch.save(policy_net.state_dict(), './dqn_save_'+str(reward_threshold)+'.pth')
            reward_threshold += 50

            #target_net.load_state_dict(policy_net.state_dict())


# env.render()
# env.close()

if __name__ == '__main__':
    main()
    print('Complete')