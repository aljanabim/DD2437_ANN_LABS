import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def mackey_glass_time_series(n_steps, step_size=1, x_0=1.5, beta=0.2, gamma=0.1, n=10, tau=25):
    x = np.zeros((n_steps))
    x[0] = x_0
    for i in range(n_steps-1):
        x_tau = x[i - tau] if i >= tau else 0
        x[i+1] = x[i] + step_size*(beta*x_tau/(1 + x_tau**n) - gamma*x[i])
    return x
