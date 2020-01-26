import torch
import numpy as np

class SequentialNet:
    def __init__(self, network_dims, learning_rate=1e-6, convergence_threshold=1e-9, min_iter=100,
                       max_iter=100000, reg_factor=0.0005):
        # Set up network dimensions
        self.network_dims = network_dims
        self.dim_in = network_dims[0]
        self.dim_hidden = network_dims[1:-1]
        self.dim_out = network_dims[-1]

        # Set up model
        device = torch.device('cpu')
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.dim_in, self.dim_hidden[0]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.dim_hidden[0], self.dim_hidden[1]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.dim_hidden[1], self.dim_out),
            ).to(device)

        self.loss = torch.nn.MSELoss(reduction='sum')

        # Set up learning hyperparameters
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.convergence_threshold = convergence_threshold
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.reg_factor = reg_factor

        # Set up bookkeeping
        self.validation_loss_record = None

    def _is_monotonically_increasing(self, values):
        increasing = True
        for i in range(len(values)-1):
            # When multiplied, True = 1 and False = 0
            increasing *= (values[i] <= values[i+1])
        return increasing

    def fit(self, train_patterns, train_targets, validation_patterns, validation_targets):
        validation_loss = np.infty
        delta_validation_loss = np.infty
        self.validation_loss_record = []

        iter = 0
        stop_flag = False
        validation_loss = np.infty
        while iter < self.max_iter and not stop_flag:
            old_validation_loss = validation_loss

            train_predictions = self.model(train_patterns)
            loss = self.loss(train_predictions, train_targets)
            self.optimizer.zero_grad()
            loss.backward()

            validation_predictions = self.model(validation_patterns)
            validation_loss = self.loss(validation_predictions, validation_targets).item()
            self.validation_loss_record.append(validation_loss/len(validation_targets))
            if iter > self.min_iter:
                if self._is_monotonically_increasing(self.validation_loss_record[-4:]):
                    stop_flag = True

            self.optimizer.step()

            # Progress bar
            if iter % int(self.max_iter/10) == 0:
                print("Training progress: {}".format(iter/self.max_iter))
                print("Iteration number {}. Average validation loss per sample: {:.2f}".format(
                    iter, validation_loss/len(validation_targets)))

            iter += 1
