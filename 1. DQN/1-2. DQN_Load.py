import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
from gym.envs.registration import register
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # training rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=695.0,
)

env = gym.make('CartPole-v5').unwrapped
N_ACTIONS = env.action_space.n # 2
N_STATES = env.observation_space.shape[0] # 4

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
print(ENV_A_SHAPE)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net = Net()
        self.target_net =  Net()
        self.train_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def select_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

dqn = DQN()
dqn.eval_net.load_state_dict(torch.load('./SaveModel/DQN_Cartpole.pth'))


def main():
    for i_episode in count(1):
        total_reward = 0
        state = env.reset()
        for t in range(700):
            env.render()
            action = dqn.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if done: # 초기 데이터가 모이지 않았을 때
                print(total_reward)
                print("end")
                break
            state = next_state

if __name__ == '__main__':
    main()
