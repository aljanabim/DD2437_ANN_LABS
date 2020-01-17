import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

def MonteCarlo(n):
    x = np.random.uniform(-1,1,n)
    y = np.random.uniform(-1,1,n)
    d = np.sqrt(np.square(x)+np.square(y))
    within = d<1
    outside = d>=1
    lenIn = list(filter(lambda x: x==True, within))
    pi = 4*len(lenIn)/len(within)
    
    
    
    plt.plot(x[within], y[within], 'g.', label=r'$\pi =$'+str(pi))
    plt.plot(x[outside], y[outside], 'r.')
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend()
    plt.show()
    
            
MonteCarlo(200000)
