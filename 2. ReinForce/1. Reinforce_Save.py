# Based on: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gym.envs.registration import register
import matplotlib.pyplot as plt

gamma = 0.99

register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=695.0,
)

env = gym.make('CartPole-v5')
env.seed(1)
torch.manual_seed(1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # nn.Linear = y=wx+b, FC(Fully Connected Layer)
        # input(state) : 1x4
        self.affine1 = nn.Linear(4, 128) # (1x4) x (4x128) = (1x128) // parameter(w) = 4 * 128
        self.affine2 = nn.Linear(128, 2) # (1x128) x (128x2) = (1x2) // parameter(w) = 128 * 2
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x) # output = 1x2 , nums of action : 2
        return F.softmax(action_scores, dim=1)

model = Policy() # 객체선언, Forward Propagation, input(state) -> Model(DNN) -> output(action)
optimizer = optim.Adam(model.parameters(), lr=1e-2) # SGD(Stochastic Gradient Decent)
eps = np.finfo(np.float32).eps.item() # 분모값이 0이 되지 않기 위해 필요한 변수

# Action을 선택
# state -> Model(DNN) -> Action -> Action에 대한 Categorical(Discrete) Distribution
# Categorical(Discrete) Distribution
# : 0이 나올 확률 0.4, 1이 나올 확률이 0.6이라고 할 때,
# argmax를 취하여 1를 선택하는 것이 아니라
# 0.4의 확률로 0이 나오게, 0.6의 확률b로 1이 나오게끔 확률을 세팅한다.

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0) # numpy to torch, +1 dimension  #dim=dim+input.dim()+1
    probs = model(state) # Action(0,1)에 대한 확률 분포 (State->Model->Action)
    m = Categorical(probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action)) # 추후 SGD 계산 편의를 위한 List
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in model.rewards[::-1]:
        R1 = R
        R = r + 0.99 * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(model.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad() # 모든 weight를 0 으로 초기화
    policy_loss = torch.cat(policy_loss).sum() # List-> (Tensor -> Cat) -> Sum
    policy_loss.backward() # SGD
    optimizer.step() # Wight Update
    del model.rewards[:]
    del model.saved_log_probs[:]
# Reward Clipping
# 3번 움직였을 때(3episode)  = (1), ( (1+1)+ + 0.99*+1) + ((1+1+1) + 0.99 * 1 +0.99^2*1)

def main():
    total_reward_list = []
    for i_episode in count(1): # while문 대용, cnt_episode 출력을 위함.
        state = env.reset() # State 초기화
        total_reward = 0
        for t in range(700):
            action = select_action(state) # Action 선택
            state, reward, done, _ = env.step(action) # 1 Step 이동 (주어진 State에서 선택한 Action을 실행)
            total_reward += reward
            #env.render()
            model.rewards.append(reward) # 정책망에 Reward 값을 저장
            if done:
                total_reward_list.append(total_reward)
                break

        finish_episode()
        if i_episode % 10 == 0: # moving, average 에피소드 10번 마다 보여주기
            print(i_episode,"번째:",total_reward_list[-5:], ", mean:", np.mean(total_reward_list[-5:], dtype=int))

        if np.mean(total_reward_list[-10:], dtype=int) > env.spec.reward_threshold: # 이동평균, 695, line 31
            torch.save(model.state_dict(),'./SaveModel/Reinforce_Save.pth')

            plt.plot(list(range(len(total_reward_list))), total_reward_list, c="b", lw=3, ls="--")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Reward Graph")
            fig = plt.gcf()
            plt.show()
            fig.savefig('./SaveModel/Reinforce_Reward_Graph.png')
            plt.close()
            break

if __name__ == '__main__':
    main()
