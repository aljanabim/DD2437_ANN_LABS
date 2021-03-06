import numpy as np
from matplotlib import pyplot as plt


class SOMNetwork():
    # 0.08 50 100
    def __init__(self, n_inputs, n_nodes, step_size=0.2, topology='linear', grid_shape=None, neighbourhood_start=50, neighbourhood_end=1, n_epochs=120, seed=False):
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_epochs = n_epochs
        self.step_size = step_size
        self.topology = topology
        self.grid_shape = grid_shape
        self.neighbourhood = neighbourhood_start
        self.neighbourhood_start = neighbourhood_start
        self.neighbourhood_end = neighbourhood_end
        self.neighbourhood_decay_rate = - \
            (1/n_epochs)*np.log(self.neighbourhood_end/self.neighbourhood_start)
        if seed:
            np.random.seed(seed)
        self.w = np.random.random((n_nodes, n_inputs))

    def fit(self, data):
        '''
        Expects data in the form
        [
            [1 0 1 0 0 1 0 1 0 1 0 0 ... 0 1] # up to 84 attributes
            [1 0 1 0 1 1 1 1 1 1 1 1 ... 1 0]
            .
            .
            .
            [1 0 1 0 1 1 1 0 0 0 0 0 ... 0 0]
            down to the number of data points

            # ROWS = n_data_points
            # COLS = n_attributes
        ]

        '''
        self.data = data
        self.n_data = data.shape[0]
        pos = np.zeros(self.n_data)
        for epoch in range(self.n_epochs):
            print('running epoch ', epoch)
            for data_row_index in range(self.n_data):
                w_distance = np.linalg.norm(
                    self.w - self.data[data_row_index, :], axis=1)
                winning_row_index = np.argmin(w_distance)

                self.update_weights(winning_row_index,
                                    self.data[data_row_index, :])
            self.neighbourhood *= np.exp(-self.neighbourhood_decay_rate)

        for row in range(self.n_data):
            w_distance = np.linalg.norm(
                self.w - self.data[row, :], axis=1)
            pos[row] = np.argmin(w_distance)
        return pos

    def update_weights(self, winning_row_index, data_row):
        neighbourhood = int(round(self.neighbourhood))
        w_height = self.w.shape[0]

        # if winning_row_index-w_height < neighbourhood:
        #     self.w[winning_row_index-neighbourhood:, :] += self.step_size * \
        #         (self.w[winning_row_index-neighbourhood:, :]-data_row)

        # self.w[winning_row_index, :] += self.step_size * \
        #     (self.w[winning_row_index, :]-data_row)
        if self.topology == 'linear':
            if winning_row_index < neighbourhood:
                self.w[0:winning_row_index+neighbourhood, :] += self.step_size * \
                    (data_row-self.w[0:winning_row_index+neighbourhood, :])

            else:
                self.w[winning_row_index-neighbourhood:winning_row_index+neighbourhood, :] += self.step_size * \
                    (data_row-self.w[winning_row_index -
                                     neighbourhood:winning_row_index+neighbourhood, :])

        if self.topology == 'circular':
            if winning_row_index < neighbourhood:
                self.w[w_height-neighbourhood+winning_row_index:, :] += self.step_size * \
                    (data_row-self.w[w_height -
                                     neighbourhood+winning_row_index:, :])
                self.w[winning_row_index:winning_row_index+neighbourhood, :] += self.step_size * \
                    (data_row -
                     self.w[winning_row_index:winning_row_index+neighbourhood, :])

            elif winning_row_index+neighbourhood > w_height-1:
                self.w[0:w_height-winning_row_index-1+neighbourhood, :] += self.step_size * \
                    (data_row-self.w[0:w_height -
                                     winning_row_index-1+neighbourhood, :])
                self.w[winning_row_index-neighbourhood:, :] += self.step_size * \
                    (data_row -
                     self.w[winning_row_index-neighbourhood:, :])

            else:
                self.w[winning_row_index-neighbourhood:winning_row_index+neighbourhood, :] += self.step_size * \
                    (data_row-self.w[winning_row_index -
                                     neighbourhood:winning_row_index+neighbourhood, :])

        if self.topology == 'grid':
            winning_matrix_row = winning_row_index // self.grid_shape
            winning_matrix_col = winning_row_index % self.grid_shape
            winning_point = np.array([winning_matrix_row, winning_matrix_col])

            for data_point_index in range(self.n_inputs):
                look_at = data_row[data_point_index]
                weight_as_matrix = self.w[:, data_point_index].reshape(
                    self.grid_shape)
                for row in range(weight_as_matrix.shape[0]):
                    for col in range(weight_as_matrix.shape[1]):
                        temp_point = np.array([row, col])
                        if int(round(np.linalg.norm(winning_point-temp_point))) <= neighbourhood:
                            weight_as_matrix[row, col] += self.step_size * \
                                (look_at-weight_as_matrix[row, col])
                self.w[:, data_point_index] = weight_as_matrix.flatten()
            # Leaky
            self.w += self.step_size*0.01 * (data_row-self.w)
