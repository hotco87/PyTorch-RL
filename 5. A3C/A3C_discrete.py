# https://github.com/MorvanZhou/pytorch-A3C/blob/master/discrete_A3C.py

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 4000

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
print(N_S, N_A)

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 100)    # 4x100
        self.pi2 = nn.Linear(100, a_dim)    # 100x2
        self.v1 = nn.Linear(s_dim, 100)     # 4x100
        self.v2 = nn.Linear(100, 1)         # 100x1
        set_init([self.pi1, self.pi2, self.v1, self.v2]) ## utils.py ## layer.wegiht(mean=0,std=0.1) ## layer.bias = 0.1
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi2(pi1)          ## (relu(4x100) * (100x2)) = 4x100x2
        v1 = F.relu(self.v1(x))         ## relu(4x100)
        values = self.v2(v1)            ## ( relu(4x100))*(100x1)) = 4x100x1
        return logits, values           ## 2, 1 (action,value)

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data    ##test.py # https://pytorch.org/docs/stable/nn.html#torch-nn-functional
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values                       # TD_loss
        c_loss = td.pow(2)                      # Critic Loss

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)            # 두 action을 distributional 하게 만듦.
        exp_v = m.log_prob(a) * td.detach()     # log_policy(N_action) * Value = (Actor loss)
                                                ## detach : 새 객체를 만들어 원래 객체의 값만 복사
                                                ## https://www.facebook.com/groups/PyTorchKR/permalink/1115798075226539/
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()   ## ??
        return total_loss

#
class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

    # main 함수 역할 (Practically)
    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset() # 상태초기화
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync, 실질적인 update
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing  ## nn. module
    opt = SharedAdam(gnet.parameters(), lr=0.0001)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(2)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()