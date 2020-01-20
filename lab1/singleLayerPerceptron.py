# %%
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(3)


def generateData(N, plot=False):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    meanA = [1, 2]
    covA = np.array([[1.2, 0],
                     [0, 0.8]])
    meanB = [-1, -2]
    covB = np.array([[0.8, 0],
                     [0, -0.5]])
    classA = np.random.multivariate_normal(meanA, covA, N)
    classB = np.random.multivariate_normal(meanB, covB, N)
    np.random.shuffle(classA)
    np.random.shuffle(classB)

    if plot:
        plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
        plt.scatter(classB[:, 0], classB[:, 1], label="Class B")
        plt.legend()
        plt.plot()
    return classA, classB


classA, classB = generateData(100, plot=True)
