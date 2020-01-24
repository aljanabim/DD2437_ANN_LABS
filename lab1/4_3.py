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

def generate_data(n_samples, offset=300):
    series = mackey_glass_time_series(n_samples+offset+5)
    patterns = []
    targets = []
    for t in range(offset, n_samples+offset):
        patterns.append([series[t-20], series[t-15], series[t-10], series[t-5], series[t]])
        targets.append(series[t+5])

    return np.array(patterns), np.array(targets)


def split_data(patterns, targets, validation_fraction=0.15, test_fraction=0.25):
    n_samples = len(patterns)
    n_train = int(n_samples*(1 - validation_fraction - test_fraction))
    n_validation = int(n_samples*validation_fraction)
    n_test = int(n_samples*test_fraction)

    train_patterns = torch.tensor(patterns[:n_train], dtype=torch.float)
    train_targets = torch.tensor(targets[:n_train], dtype=torch.float)
    validation_patterns = torch.tensor(patterns[n_train:(n_train+n_validation)], dtype=torch.float)
    validation_targets = torch.tensor(targets[n_train:(n_train+n_validation)], dtype=torch.float)
    test_patterns = torch.tensor(patterns[-n_test:], dtype=torch.float)
    test_targets = torch.tensor(targets[-n_test:], dtype=torch.float)

    return train_patterns, train_targets, validation_patterns, validation_targets, test_patterns, test_targets




device = torch.device('cpu')

max_iter = 10000
n_samples = 1200
dim_hidden_1 = 5

patterns, targets = generate_data(n_samples)
train_patterns, train_targets, validation_patterns, validation_targets,  test_patterns, test_targets = split_data(
    patterns, targets, validation_fraction=400/1200, test_fraction=200/1200)

dim_in = train_patterns.shape[1]
dim_out = 1

weights_1 = torch.randn(dim_in, dim_hidden_1, requires_grad=True)
weights_2 = torch.randn(dim_hidden_1, dim_out, requires_grad=True)

learning_rate = 1e-7
convergence_threshold = 1e-8
train_loss = np.infty
delta_train_loss = np.infty

iter = 0
validation_loss_record = []
while delta_train_loss > convergence_threshold and iter <= max_iter:
    old_train_loss = train_loss

    train_predictions = torch.nn.functional.sigmoid(train_patterns.mm(weights_1)).mm(weights_2)
    train_loss = (train_predictions - train_targets).pow(2).sum()
    train_loss.backward()

    with torch.no_grad():
        weights_1 -= learning_rate * weights_1.grad
        weights_2 -= learning_rate * weights_2.grad
        weights_1.grad.zero_()
        weights_2.grad.zero_()

    if iter % int(max_iter/10) == 0:
        print(iter/max_iter)
        print(iter, train_loss.item())

    delta_train_loss = abs(old_train_loss - train_loss)
    iter += 1
    with torch.no_grad():
        validation_predictions = torch.nn.functional.sigmoid(validation_patterns.mm(weights_1)).mm(weights_2)
        validation_loss = (validation_predictions - validation_targets).pow(2).sum().item()
        validation_loss_record.append(validation_loss)

print(iter)
print(len(validation_loss_record))
plt.plot(validation_loss_record[20:])
plt.show()



test_predictions = torch.nn.functional.relu(test_patterns.mm(weights_1)).mm(weights_2)
test_loss = (test_predictions - test_targets).pow(2).sum()
print("Test loss: {}".format(test_loss))


plt.plot(test_predictions.detach().numpy(), label='Prediction')
plt.plot(test_targets.detach().numpy(), label='Target')
plt.legend()
plt.show()
