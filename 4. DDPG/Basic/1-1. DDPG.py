import numpy as np
import random


transition = np.hstack((1, [2, 3], 4, 5,6,7,8,9,99))
transition1 = []

for i in range(5):
    transition1.append((i, [i+1, i+2], i+3, i+4))

print(transition1)
print(transition1[0])

print(random.sample(transition1, 1))
print(transition)
print(np.random.choice(transition, 2))
#print(transition1)


target = [1,2,3,4]
source = [5,6,7,8,]
# for t, s in target, source:
#     print(t)
#     print(s)

# for t in target:
#     print(t)
    #print(s)

for x,y in zip (target,source):
    print(x)
    print(y)

#
# for t, s in target, source:
#     print(target)
#     print(source)

