# %%
import numpy as np


def double(x, center):
    return x*2+center


vfunc = np.vectorize(double)

x = np.array([np.arange(0, 10, 1)]).T
y = vfunc(x, [0.5, 0.7])
print(y)
