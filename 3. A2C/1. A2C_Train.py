# Based on: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

gamma = 0.99
torch.manual_seed(450)
env = gym.make('CartPole-v1')
env.seed(450)
state_size = env.observation_space.shape[0] # 4
action_size = env.action_space.n # 2
print(action_size)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # nn.Linear = FC(Fully Connected Layer, y=wx+b)
        self.affine1 = nn.Linear(state_size, 128) # (1x4) x (4x128) = (1x128) ,  # of parameters(w) = 4 * 128
        self.action_head = nn.Linear(128, 2) # actor, output = (1x2)
        self.value_head = nn.Linear(128, 1) # critic, output = (1x1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x) # actor (1x2) , torch.Size([2])
        state_values = self.value_head(x) # critic (1x1), torch.Size([2])
        return F.softmax(action_scores, dim=-1), state_values # (,2), (1x1), # dim=-1. 가장 안쪽 행렬의 요소 계산(행)

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item() # 분모값이 0이 되지 않기 위해 필요한 변수

# saved.actions 값에 action_log_prob(액션로그확률), state_value 저장(추후 계산(cost, GD)을 쉽게 하기 위함.)
def select_action(state):
    state = torch.Tensor(state)         # torch.from_numpy(state).float()
    probs, state_value = model(state)   # model을 통과한 return 값 (action, state_value)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item() # action 선택


def finish_episode():
    R = 0 # init , Gt(utility)
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]: # [::-1] 행렬의 순서를 바꾸어 출력하는 기능
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) # Reward Clipping
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item() # Base line, final Gt
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]))) # huber loss
    optimizer.zero_grad() #  모든 weight를 0 으로 초기화
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    total_reward_list = []
    for i_episode in count(1):
        total_reward = 0
        state = env.reset()
        for t in range(700):
            action = select_action(state)               # state(input) -> model -> action(output1), state_value(output2)
            state, reward, done, _ = env.step(action)   # reward 1 or 0
            total_reward += reward
            model.rewards.append(reward)
            if done:
                total_reward_list.append(total_reward)
                break
        finish_episode() # training

        if i_episode % 10 == 0: # 에피소드 10번 마다 보여주기
            print(i_episode,"번째:",total_reward_list[-5:], ", mean:", np.mean(total_reward_list[-5:], dtype=int))

        if np.mean(total_reward_list[-10:], dtype=int) > env.spec.reward_threshold:
            torch.save(model.state_dict(),'./SaveModel/actor_critic_save.pth')

            plt.plot(list(range(len(total_reward_list))), total_reward_list, c="b", lw=3, ls="--")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Reward Graph")
            fig = plt.gcf()
            plt.show()
            fig.savefig('./SaveModel/reward_graph.png')
            plt.close()
            break

if __name__ == '__main__':
    main()
