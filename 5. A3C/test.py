import torch.nn.functional as F
# import torch.distributions.categorical
from torch.distributions.categorical import Categorical
import torch
import math
#import torch.distributions.Categorical
from torch import nn


# --------------------------------------------------#
# torch.nn.functional.softmax
logits = [[1,2,3]]
logits_T = torch.Tensor(logits) # float32(.dtype)
print('logit_T  ',logits_T)
prob = F.softmax(logits_T, dim=1).data
prob1 = F.softmax(logits_T)
print(prob)
print(prob1)
print(1e1/(1e1+1e2+1e3),'/t', 2e1/(1e1+1e2+1e3), '/t',3e1/(1e1+1e2+1e3))
# --------------------------------------------------#


# --------------------------------------------------#
# # torch.distributions import Categorical
# a = torch.tensor([ 0.25, 0.25, 0.25, 0.25 ])
# m = Categorical(a)
# print(m.sample())  # equal probability of 0, 1, 2, 3
# print(m.log_prob(a))
# print(math.log(0.25))

# logits = [[0.45,0.55]]
# logits_T = torch.Tensor(logits)
# prob = F.softmax(logits_T, dim=1).data
# print(prob)
# m = Categorical(prob) # action 둘 중 하나 선택
# for i in range(5):
#     print(m.sample())  # equal probability of 0, 1, 2, 3
#     # print(m.sample().numpy())
#     # print(m.sample().numpy()[0])
# --------------------------------------------------#


# --------------------------------------------------#
# # init normalization
# w = torch.empty(3, 3) # torch.tensor([])
# print(nn.init.normal_(w,mean=0,std=1))   # mean = 0, std =1
# print(nn.init.constant_(w, 0.1))
# --------------------------------------------------#