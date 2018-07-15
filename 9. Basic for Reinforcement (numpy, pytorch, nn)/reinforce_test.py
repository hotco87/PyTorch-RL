import torch
from torch.distributions.distribution import Distribution
from torch.distributions.multinomial import Multinomial
from torch.distributions import Categorical
import torch
import numpy as np

# Referencef
# https://pytorch.org/docs/stable/distributions.html
# https://pytorch.org/docs/stable/_modules/torch/distributions/multinomial.html

print("-----------------------------------------")
print('Multinomial Distribution\n')

m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
print(m.sample())  # equal probability of 0, 1, 2, 3

m = Categorical(torch.tensor([ 0.8, 0.15, 0.05]))
sample = m.sample()
print(sample)
print(type(sample))  # equal probability of 0, 1, 2, 3
print(m.log_prob(sample))

probs_2d = [0.1, 0.9]
probs_2d = torch.Tensor(probs_2d)
sample_2d = torch.multinomial(probs_2d, 1, True)
print(sample_2d)

# Multinomial = (100, torch.tensor([1, 1]))
# x = m.sample()
# print(x)
# print(type(x))
# Multinomial(torch.tensor([1., 1., 1., 1.])).log_prob(x)

# https://pytorch.org/docs/stable/distributions.html#categorical
print("-----------------------------------------\n")


#
#
# print("-----------------------------------------")
# print('Torch Cat Sum\n')
# torch.manual_seed(999)
# print("##########  Input   ##########")
# x = torch.randn(2, 2)
# y = torch.randn(2, 2)
# print(x)
# print(y)
# print("##########  torch cat   ##########")
# z = torch.cat((x,y))
# print(z)
# z = torch.cat((x,y),0)
# print(z)
# z = torch.cat((x,y),1)
# print(z)
# print("##########  torch cat sum   ##########")
# z = torch.cat((x,y)).sum()
# print(z)
# print("-----------------------------------------")

# a.append(torch.randn(1))
# a.append(torch.randn(1))
# a.append(torch.randn(1))
# print(a)
# print(np.shape(a))
# # print(torch.sum(a)) #<-- 계산이 안됨.
# print(torch.cat(a).sum())



