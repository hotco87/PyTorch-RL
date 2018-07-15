import numpy as np

a =range(5)
print(a)

a = list(range(5))
print(a)

b = a[-3:]
print(b)
print(np.mean(b,dtype=int))
