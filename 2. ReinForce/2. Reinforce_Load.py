import gym
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gym.envs.registration import register

register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=695.0,
)

env = gym.make('CartPole-v5')


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x) # output = 1x2 , nums of action : 2
        return F.softmax(action_scores, dim=1)

model = Policy() # Forward Propagation, input(state) -> Model(DNN) -> output(action)
model.load_state_dict(torch.load('./SaveModel/Reinforce_Save.pth'))

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0) # +1 dimension  #dim=dim+input.dim()+1
    probs = model(state) # Action(0,1)에 대한 확률 분포 (State->Model->Action)
    m = Categorical(probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action)) # 추후 SGD 계산 편의를 위한 List
    return action.item()

def main():
    for i_episode in count(1): # while문 대용, cnt_episode 출력을 위함.
        state = env.reset() # State 초기화
        total_reward = 0
        for t in range(700):
            env.render()
            action = select_action(state) # Action 선택
            state, reward, done, _ = env.step(action) # 1 Step 이동 (주어진 State에서 선택한 Action을 실행)
            total_reward += reward

            if done:
                env.close()
                print(total_reward)
                break

        env.close()

if __name__ == '__main__':
    main()
