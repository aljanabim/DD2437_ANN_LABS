import numpy as np


class RBFNetwork():
    def __init__(self,
                 n_inputs,
                 n_rbf,
                 n_outputs,
                 n_epochs=100,
                 learning_rate_start=0.1,
                 learning_rate_end=0.001,
                 cl_learning_rate = 0.2,
                 cl_leak_rate = 0.001,
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
        self.cl_learning_rate = cl_learning_rate
        self.cl_leak_rate = cl_leak_rate

        self.learning_rate_record = []
        self.mse_record = None

    def _base_func(self, x, center):
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * self.rbf_var**2))

    def fit(self, data, f, method='batch', cl_method=None):
        self.mse_record = []
        self.data = np.array([data]).T
        if method == 'batch':
            phi = self.RBF(self.data, self.rbf_centers)
            self.w = np.dot(
                np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)
        elif method == "sequential":
            for _ in range(self.n_epochs):
                for k, x_k in enumerate(data):
                    if cl_method == "basic":
                        self._cl_step(x_k, leaky=False)
                    elif cl_method == 'leaky':
                        self._cl_step(x_k, leaky=True)
                    self.w += self._calc_delta_w(x_k, f[k])

                self.learning_rate *= np.exp(-self.learning_rate_decay)
                self.learning_rate_record.append(self.learning_rate)
                preds = self.predict(data)
                self.mse_record.append(self.calc_mse(preds, f))

    def _calc_delta_w(self, pattern, target):
        error = target - self.predict(pattern)
        delta_w = self.learning_rate * error * \
            self.RBF(pattern, self.rbf_centers)
        return delta_w

    def _cl_step(self, pattern, leaky=False):
        rbf_values = self.RBF(pattern, self.rbf_centers)
        winner_index = np.argmax(rbf_values)
        winner_center = self.rbf_centers[0, winner_index]
        self.rbf_centers[0, winner_index] += self.cl_learning_rate*(pattern - winner_center)
        if leaky:
            self.rbf_centers[0, :] += self.cl_leak_rate*(pattern - self.rbf_centers[0, :])

    def calc_mse(self, preds, targets):
        return np.sum(np.power(preds - targets, 2))/len(targets)


    def predict(self, x):
        x = np.array([x]).T
        return np.dot(self.w, self.RBF(x, self.rbf_centers).T).flatten()
