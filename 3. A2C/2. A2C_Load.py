import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

env = gym.make('CartPole-v1')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2) # actor
        self.value_head = nn.Linear(128, 1) # critic

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x) # actor (1X2)
        state_values = self.value_head(x) # critic (1x1)
        return F.softmax(action_scores, dim=-1), state_values # (,2), (1x1)

model = Policy()
model.load_state_dict(torch.load('./SaveModel/actor_critic_save.pth'))

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state) # policy, state_value
    m = Categorical(probs)
    action = m.sample()
    return action.item()

def main():
        state = env.reset()
        for t in range(10000):
            env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            print(t)
            if done:
                env.close()
                break

        env.close()

if __name__ == '__main__':
    main()
    env.close()
