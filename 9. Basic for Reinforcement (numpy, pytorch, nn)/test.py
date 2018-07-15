import visdom
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# vis = visdom.Visdom()
# vis.text('Hello, world!')
# vis.image(np.ones((20, 20, 20)))


# plt.figure(2)
# plt.clf()
# durations_t = torch.tensor(episode_durations, dtype=torch.float)
# plt.title('Training...')
# plt.xlabel('Episode')
# plt.ylabel('Duration')
# plt.plot(durations_t.numpy())

# plt.plot([10, 20, 30, 40], [1, 4, 9, 16], c="b",
#          lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r")
# plt.xlabel("Epsidoe")
# plt.ylabel("Reward")
# plt.title("Reward Graph")
# plt.show()
# plt.savefig('books_read.png')

# plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
# plt.show()
# plt.savefig('./books_read.png')
#
# plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
# plt.xlabel('Distance in meters')
# plt.ylabel('Gravitational force in newtons')
# plt.title('Gravitational force and distance')
#
# fig = plt.gcf()  # 변경한 곳
# plt.show()
# a= 4
# fig.savefig('./dqn_save'+str(a)+'.png')
# #fig.savefig('GG.png')


#print(100000000000000000**10000000000000)

# import torch
# # a = torch.Tensor([10000.])
# # b = torch.Tensor([2222222222100000000000000000000000000000000000000000000000000000000000.])
# # print(a/b)
#
#
# x = torch.randn(2, 1)
# print(x)
# print('\n')
# y = torch.randn(2, 1)
# print(y)
# print('\n')
# print(torch.stack([x]))
# print('\n')
# print(torch.stack([y]))
# #print(torch.stack([x,y]))
# sum = [torch.stack([x]).sum()]
# print(sum)
# print(np.shape(sum))

import random
print(np.random.rand())
print(np.random.uniform())

import matplotlib.pyplot as plt
s = np.random.uniform()
print(s)
s1 = np.random.randrand(0,1,1000)
count, bins, ignored = plt.hist(s, 15, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

# count, bins, ignored = plt.hist(s1, 15, normed=True)
# plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()
