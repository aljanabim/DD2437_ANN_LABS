import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

class FullyConnectedNet:
    def __init__(self, network_dims, learning_rate=1e-6, convergence_threshold=1e-9, min_iter=100,
                       max_iter=100000, reg_factor=0.0005):
        # Set up network dimensions
        self.network_dims = network_dims
        self.dim_in = network_dims[0]
        self.dim_hidden = network_dims[1:-1]
        self.dim_out = network_dims[-1]

        # Set up weights
        self.weights_in = torch.randn(self.dim_in+1, self.dim_hidden[0], requires_grad=True)
        self.weights_hidden = []
        for i in range(len(self.dim_hidden)-1):
            self.weights_hidden.append(
                torch.randn(self.dim_hidden[i]+1, self.dim_hidden[i+1], requires_grad=True))
        print(len(self.weights_hidden))
        self.weights_out = torch.randn(self.dim_hidden[-1]+1, self.dim_out, requires_grad=True)
        self.weights_all = [self.weights_in, *self.weights_hidden, self.weights_out]

        # Set up learning hyperparameters
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.reg_factor = reg_factor

        # Set up bookkeeping
        self.validation_loss_record = None

    def reset_weights(self, input):
        for weights in weights_all:
            weights.randn_()
            weights.grad.zero_()

    def extend_with_bias(self, matrix):
        """Add column of ones to matrix, in order to accomodate bias weights."""
        n_rows = matrix.shape[0]
        ones = torch.ones((n_rows, 1))
        return torch.cat((matrix, ones), 1)

    def forward(self, input):
        # Input layer
        x = torch.sigmoid(torch.mm(self.extend_with_bias(input), self.weights_in))
        # Hidden layers
        for weights in self.weights_hidden:
            x = torch.sigmoid(torch.mm(self.extend_with_bias(x), weights))
        # Output layer
        output = torch.mm(self.extend_with_bias(x), self.weights_out)

        return output

    def loss(self, predictions, targets):
        return (predictions - targets).pow(2).sum()

    def l1_regularizer(self, weights_set):
        loss = 0
        for weights in weights_set:
            loss += weights.abs().sum()
        return loss

    def l2_regularizer(self, weights_set):
        loss = 0
        for weights in weights_set:
            loss += weights.pow(2).sum()
        return loss

    def reg_loss(self, weights_set, mode=None):
        if not mode:
            return 0
        elif mode == "l1":
            return self.l1_regularizer(weights_set)
        elif mode == "l2":
            return self.l2_regularizer(weights_set)

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

        # while delta_validation_loss > self.convergence_threshold and iter < self.max_iter:
        iter = 0
        stop_flag = False
        while iter < self.max_iter and not stop_flag:
            old_validation_loss = validation_loss

            train_predictions = self.forward(train_patterns)
            train_loss = (self.loss(train_predictions, train_targets)
                          + self.reg_factor * self.reg_loss(self.weights_all, mode='l2'))
            train_loss.backward()

            with torch.no_grad():
                for weights in self.weights_all:
                    weights -= self.learning_rate*weights.grad
                for weights in self.weights_all:
                    weights.grad.zero_()

                validation_predictions = self.forward(validation_patterns)
                validation_loss = self.loss(validation_predictions, validation_targets).item()
                self.validation_loss_record.append(validation_loss/len(validation_targets))
                if iter > self.min_iter:
                    if self._is_monotonically_increasing(self.validation_loss_record[-4:]):
                        stop_flag = True

            # Progress bar
            if iter % int(self.max_iter/10) == 0:
                print("Training progress: {}".format(iter/self.max_iter))
                print("Iteration number {}. Average validation loss per sample: {:.2f}".format(
                    iter, validation_loss/len(validation_targets)))

            iter += 1

    def fit_with_torch_optimizer(self, train_patterns, train_targets, validation_patterns,
                                 validation_targets):
        self.validation_loss_record = []
        optimizer = torch.optim.Adam(self.weights_all, lr=self.learning_rate)

        stop_flag = False
        iter = 0
        while iter < self.max_iter and not stop_flag:
            # Training
            train_predictions = self.forward(train_patterns)
            loss = (self.loss(train_predictions, train_targets)
                    + self.reg_factor * self.reg_loss(self.weights_all, mode='l2'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            with torch.no_grad():
                validation_predictions = self.forward(validation_patterns)
                validation_loss = float(self.loss(validation_predictions, validation_targets).item())
                self.validation_loss_record.append(validation_loss/len(validation_targets))
                if iter > self.min_iter:
                    if self._is_monotonically_increasing(self.validation_loss_record[-4:]):
                        stop_flag = True

            # Progress bar
            if iter % int(self.max_iter/10) == 0:
                print("Training progress: {}".format(iter/self.max_iter))
                print("Iteration number {}. Average validation loss per sample: {:.2f}".format(
                    iter, validation_loss/len(validation_targets)))

            iter += 1
