import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # exploration
GAMMA = 0.9                 # reward discount factor
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=630.0,
)

env = gym.make('CartPole-v5').unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # weight initialization : mean =0, std =1 , normal distribution
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

    def select_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)

        if np.random.uniform() < EPSILON:   # Exploration
            actions_value = self.eval_net.forward(x) # [[0.3, 0.9]]
            action = torch.max(actions_value, 1)[1].data.numpy() # 큰 값의 인덱스를 뽑아냄 [1], (0.9 > 0.3)
            action = action[0] # 1
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action # 0과 1을 이용

    def replay_memory(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # 4, 1, 1, 4 = size(10)
        index = self.memory_counter % MEMORY_CAPACITY # 2000step 마다 각각 메모리 갱신
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train(self):
        # target parameter update
        if self.train_step_counter % TARGET_REPLACE_ITER == 0: # 100step 마다 eval_net -> target_network (복사)
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.train_step_counter += 1

        # batch 사이즈 만큼 transitions 을 뽑아냄.
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # 메모리에서 랜덤 샘플링 할 index 추출, 32
        memory_batch = self.memory[sample_index, :] # 메모리 추출, (32,10), 32:배치사이즈, 10: s(4),a(1),r(1),s(4)
        state_batch = torch.FloatTensor(memory_batch[:, :N_STATES]) #(32,4)
        action_batch = torch.LongTensor(memory_batch[:, N_STATES:N_STATES+1].astype(int)) # (32,1) : [[0],[1],[1],[0],....], (중의적으로 index의 의미를 가짐.)
        reward_batch = torch.FloatTensor(memory_batch[:, N_STATES+1:N_STATES+2]) #(32,1)
        next_state_batch = torch.FloatTensor(memory_batch[:, -N_STATES:]) #(32,4)

        # cost function
        q_eval = self.eval_net(state_batch).gather(1, action_batch) # 실제 q값을 뽑아냄, [32,1], action batch(index)를 이용
        q_next = self.target_net(next_state_batch).detach() # detach : 훈련할 때, q_next는 backprob을 하지 않음
        q_target = reward_batch + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = nn.MSELoss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()


def main():
    total_reward_list = []
    for i_episode in count(1):
        total_reward = 0
        state = env.reset()
        for t in range(700):
            #env.render()
            action = dqn.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # reward shaping
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 # 카트가 중간에 있을 수록 좋음.(카트와 중심 사이의 거리를 작게 하면 보상이 커짐.)
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # 막대기의 각도가 90도에 가까울 수록 보상이 커짐.
            reward = r1 + r2

            dqn.replay_memory(state, action, reward, next_state) # 매 step 마다 메모리에 저장 (s,a,r,s')

            #print(dqn.memory_counter)
            if dqn.memory_counter > MEMORY_CAPACITY: # 메모리가 2000이 다 채워졌을 때, train 시작 # frameskip은 내장되어 있음.
                dqn.train()
                if done:
                    total_reward_list.append(total_reward)
                    break

            if done: # 초기 데이터가 모이지 않았을 때
                break
            state = next_state

        # evaluation
        if i_episode % 10 == 0:
            print(i_episode, "번째:", total_reward_list[-5:], ", mean:", np.mean(total_reward_list[-5:], dtype=int))

        if np.mean(total_reward_list[-20:], dtype=int) > env.spec.reward_threshold:
            torch.save(dqn.eval_net.state_dict(), './SaveModel/DQN_Cartpole.pth')
            plt.plot(list(range(len(total_reward_list))), total_reward_list, c="b", lw=3, ls="--")
            plt.xlabel("Episode"), plt.ylabel("Reward"), plt.title("Reward Graph")
            fig = plt.gcf()
            plt.show()
            fig.savefig('./SaveModel/DQN_Reward_Graph.png')
            plt.close()

        if np.mean(total_reward_list[-20:], dtype=int) > env.spec.reward_threshold+5:
            break

if __name__ == '__main__':
    main()