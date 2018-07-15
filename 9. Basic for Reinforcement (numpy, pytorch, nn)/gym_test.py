import gym
from gym.envs.registration import register
from itertools import count

register(
    id='CartPole-v5',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 700},
    reward_threshold=695.0,
)

env = gym.make('CartPole-v5')

print(env.spec.reward_threshold)
print(type(env.spec.tags))

env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 650
print(env.spec.tags) # dic
print(env.spec.tags.keys())
print(env.spec.tags.values())

for t in count():
    print(t)