import numpy as np


class HopfieldNetwork():
    def __init__(self, n_nodes):
        self.w = None
        self.n_nodes = n_nodes
        self.nodes = np.zeros(self.n_nodes)

    def fit(self, data):
        '''
        Expects data as rows of patters
        '''
        self.data = data
        self.len_pattern = self.data.shape[0]
        self.n_patterns = self.data.shape[1]

        for pattern_index in range(self.n_patterns):
            self.w[pattern_index] =

        # sequential update
        for i in range():

        self.w =
