import numpy as np



def load_data():
    with  open('lab3/data/pict.dat', 'r') as f:
        text = str(f.read()).split(',')
        value_list = np.array([int(val) for val in text])
        patterns = []
        for n in range(11):
            start_index = 1024*n
            end_index = 1024*(n+1)
            patterns.append(value_list[start_index:end_index])
        return np.array(patterns)
