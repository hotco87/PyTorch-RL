import sys
import gym
import torch
import random
import numpy as np
from collections import deque
from datetime import datetime
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
MEMORY_CAPACITY = 50000


class OrnsteinUhlenbeckActionNoise(object): # Select Action Exploration (Noise)
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
        self.memory_counter = 0
        self.ou = OrnsteinUhlenbeckActionNoise(theta=ou_theta, sigma=ou_sigma, mu=ou_mu, action_dim=N_ACTIONS) # explortion


    def get_action(self, state):
        state = torch.from_numpy(state).float() # numpy -> torch -> float
        model_action = self.actor(state).detach().numpy() * R_ACTIONS # 1step(모델)을 지나가도 다른 그래프에 영향을 주지 않음. 새 객체를 만들어 원래 객체의 값만 복사
        action = model_action + self.ou.sample() * R_ACTIONS
        return action

    def update_target_model(self):  # soft_update
        target = [self.actor_target, self.critic_target]
        source = [self.actor,self.critic]
        for t,s in zip(target,source) :
            for target_param, param in zip(t.parameters(), s.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def replay_memory(self, s, a, r, s_, d):
        self.memory.append((s, a, r, s_, d))
        self.memory_counter += 1

    def _get_sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def train(self):
        memory_batch = random.sample(self.memory, BATCH_SIZE)
        memory_batch = np.array(memory_batch).transpose()

        states = torch.from_numpy(np.vstack(memory_batch[0])).float()            # (64,3)
        actions = torch.from_numpy(np.vstack(memory_batch[1])).float()           # (64,1)
        rewards = torch.from_numpy(np.vstack(memory_batch[2])).float()           # (64,1)
        next_states = torch.from_numpy(np.vstack(memory_batch[3])).float()       # (64,3)
        dones = torch.from_numpy(np.vstack(memory_batch[4].astype(int))).float() # 64

        # critic update
        self.critic_optimizer.zero_grad()
        value = self.critic(states, actions)                         # Main Net             -> transition의 (state,action) 을 critic 통과
        next_actions = self.actor_target(next_states)                # Target_Net action    -> next_state 를 통과하여 action 예측
        next_value = self.critic_target(next_states, next_actions)   # Target_Net Q_value   -> next_state, next_action 으로 부터 critic1, 2 통과
        # next_actions, next_value, next_value 는 새롭게 구함.
        # s,a,r,s'  = s,a   -> v (cur_value)
        # s'      ->  actor_target   -> a'(next_actions)
        # s', a'  ->  critic_target  -> v'(next_value)
        # s,a,r,s'  = s',a' -> v'(next_value)
        target = rewards + (1 - dones) * GAMMA * next_value          # reward + gamma *V'
        critic_loss = F.mse_loss(value, target)                      # (reward + gamma *V' - v)^2
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        pred_actions = self.actor(states)                       # s -> actor -> a        # for back_prob, action을 다시 예측
        actor_loss = self.critic(states, pred_actions).mean()   # s,a -> critic -> value
        actor_loss = -actor_loss
        actor_loss.backward()
        self.actor_optimizer.step()

ddpg = DDPG()

def main():
    total_reward_list = []
    finish_reward = -130
    for i_episode in count(1):
        total_reward = 0
        state = env.reset()   # [ , , ] = (3,)

        for t in range(700):
            action = ddpg.get_action(state)
            next_state, reward, done, info = env.step(action)
            reward = float(reward)
            ddpg.replay_memory(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if ddpg.memory_counter > BATCH_SIZE: # BATCH_SIZE(64) 이상일 때 부터 train 시작
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
            print("save model start")
            torch.save(ddpg.actor.state_dict(), './SaveModel/DDPG_actor.pth')
            plt.plot(list(range(len(total_reward_list))), total_reward_list, c="b", lw=3, ls="--")
            plt.xlabel("Episode"), plt.ylabel("Reward"), plt.title("Reward Graph")
            fig = plt.gcf()
            fig.savefig('./SaveModel/DDPG_Reward_Graph.png')
            plt.close()

            if np.mean(total_reward_list[-20:]) > (finish_reward+40):
                print("finish")
                torch.save(ddpg.actor.state_dict(), './SaveModel/DDPG_actor_finished.pth')
                break

if __name__ == '__main__':
    main()