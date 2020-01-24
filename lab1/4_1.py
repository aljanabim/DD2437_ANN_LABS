import numpy as np
import matplotlib.pyplot as plt
import torch

plt.style.use('ggplot')

def mackey_glass_time_series(n_steps, step_size=1, x_0=1.5, beta=0.2, gamma=0.1, n=10, tau=25):
    x = np.zeros((n_steps))
    x[0] = x_0
    for i in range(n_steps-1):
        x_tau = x[i - tau] if i >= tau else 0
        x[i+1] = x[i] + step_size*(beta*x_tau/(1 + x_tau**n) - gamma*x[i])
    return x

device = torch.device('cpu')

n_samples = 4
dim_in = 1
dim_out = 1
dim_hidden_1 = 4

weights_1 = torch.randn(dim_in, dim_hidden_1, requires_grad=True)
weights_2 = torch.randn(dim_hidden_1, dim_out, requires_grad=True)

for t in range(500):
    y_pred = x.mm(weights_1).clamp(min=0).mm(weights_2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    loss.backward()

    with torch.no_grad():
        weights_1 -= learning_rate * weights_1.grad
        weights_2 -= learning_rate * weights_2.grad

    weights_1.zero_()
    weights_2.zero_()
