# %%
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(3)

class Perceptron():
    def __init__(self, learning_method="perceptron", learning_rate=0.1):
        # Learning parameters
        self.learning_method = learning_method
        self.learning_rate = learning_rate
        # Model variables
        self.W = None
        self.n_inputs = None
        self.n_outputs = None
        

    def predict(self, x):
        if self.W.dot(x) >= 0:
            return 1
        else:
            return -1

    def fit(self, data, labels):        
        if self.learning_method == "perceptron":
            self.perceptron_fit(data, labels)
        elif self.learning_method == "delta":
            self.delta_fit(data, labels)
        

    def perceptron_fit(self, data, labels):
        #TODO: Implement
        

    def delta_fit(self, data, labels):
        self.n_inputs = len(data)
        self.W = np.random(-1, 1, (1, self.n_inputs))
        delta_W = np.zeros((1, self.n_inputs))
        delta_W = -self.learning_rate * data @ ( )
        
        

        #TODO: Implement


def generate_data(N, plot=False):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    meanA = [1, 3]
    covA = np.array([[1.2, 0],
                     [0, 0.8]])
    meanB = [-1, -2]
    covB = np.array([[0.8, 0],
                     [0, -0.5]])
    classA = np.random.multivariate_normal(meanA, covA, N)
    classB = np.random.multivariate_normal(meanB, covB, N)
    data = np.array([[classA, np.zeros(N)],
                     [classB, np.ones(N)]])
    if plot:
        plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
        plt.scatter(classB[:, 0], classB[:, 1], label="Class B")
        plt.legend()
        plt.plot()
    
    return classA, classB

hejhej
classA, classB = generateData(100, plot=True)
