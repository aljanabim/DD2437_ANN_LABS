import numpy as np


class RBFNetwork():
    def __init__(self,
                 n_inputs,
                 n_rbf,
                 n_outputs,
                 n_epochs=100,
                 learning_rate_start=0.1,
                 learning_rate_end=0.001,
                 min_val=0.05,
                 max_val=2 * np.pi,
                 rbf_var=0.1):
        self.n_inputs = n_inputs
        self.n_rbf = n_rbf
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.rbf_var = rbf_var

        self.rbf_centers = np.array([np.linspace(min_val, max_val, n_rbf)])
        self.w = np.array([np.random.normal(0, 1, n_rbf)])
        self.RBF = np.vectorize(self._base_func)
        self.learning_rate_decay = -(1/n_epochs) * ( np.log(learning_rate_end) - np.log(learning_rate_start) )
        self.learning_rate = learning_rate_start
        self.learning_rate_record = []

    def _base_func(self, x, center):
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * self.rbf_var**2))

    def fit(self, data, f, method='batch'):
        self.data = np.array([data]).T

        if method == 'batch':
            phi = self.RBF(self.data, self.rbf_centers)
            self.w = np.dot(
                np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)
        if method == "sequential":
            for _ in range(self.n_epochs):
                for k, x_k in enumerate(data):
                    # temp_learning_rate = self.learning_rate/np.sqrt(k)
                    # print(temp_learning_rate)
                    error = f[k] - self.predict(x_k)
                    delta_w = self.learning_rate * error * \
                        self.RBF(x_k, self.rbf_centers)
                    self.w += delta_w
                self.learning_rate *= np.exp(-self.learning_rate_decay)
                self.learning_rate_record.append(self.learning_rate)

    def predict(self, x):
        x = np.array([x]).T
        return np.dot(self.w, self.RBF(x, self.rbf_centers).T).flatten()
