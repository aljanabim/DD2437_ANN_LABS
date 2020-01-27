import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def generate_data(N, plot=False):
    '''
    Generates data of two linearly seperable classes of N samples
    '''
    data= np.ones((N,8))*(-1)
    
    for i in range(N):
        data[i,np.random.randint(0,8)] =1 
        
    print(data)

    # if plot:
    #     plt.scatter(classA[:, 0], classA[:, 1], label="Class A")
    #     plt.scatter(classB[:, 0], classB[:, 1], label="Class B")

    #     plt.plot()
    return data

if __name__== "__main__":
    generate_data(10)