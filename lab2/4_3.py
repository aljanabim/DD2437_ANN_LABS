import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pylab import *
import matplotlib.patches as mpatches
from som_net import SOMNetwork
import re
# np.set_printoptions(threshold=sys.maxsize)
# plt.style.use('ggplot')
plt.grid(False)
plt.axis('off')

PARTIES_MAP = {
    0: None,
    1: 'm',
    2: 'fp',
    3: 's',
    4: 'v',
    5: 'mp',
    6: 'kd',
    7: 'c'
}
SEX_MAP = {
    0: 'male',
    1: 'female'
}
COLORS_PARTIES = ['#ffffff', '#4363bd', '#911eb4',
                  '#e6194b', '#f58231', '#bfef45', '#42d4f4', '#3cb44b']
COLORS_SEX = ['#42d4f4', '#FF0084']

PARTY_COLORS = {
    0: '#333',
    1: '#4363bd',
    2: '#911eb4',
    3: '#e6194b',
    4: '#f58231',
    5: '#bfef45',
    6: '#42d4f4',
    7: '#3cb44b',
}

SEX_COLORS = {
    0: '#42d4f4',
    1: '#FF0084'
}


def data():
    f_votes = open('./data/votes.dat', 'r').readlines()[0].split(',')
    votes = np.array([f_votes]).reshape(349, 31).astype('float')

    f_parties = open('./data/mpparty.dat', 'r').readlines()[3:]
    parties = np.array([party.split('\n')[0]
                        for party in f_parties]).astype('int')

    f_sex = open('./data/mpsex.dat', 'r').readlines()[2:]
    sex = np.array([sex.split('\n')[0] for sex in f_sex]).astype('int')

    f_districts = open('./data/mpdistrict.dat', 'r').readlines()
    districts = np.array([district.split('\n')[0]
                          for district in f_districts]).astype('int')
    f_names = open('./data/mpnames.txt', 'r').readlines()
    names = np.array([name.split('\n')[0] for name in f_names])

    return votes, parties, sex, districts, names


votes, parties, sex, districts, names = data()


def contains_duplicates(X):
    return len(np.unique(X)) != len(X)


def r(): return np.random.randint(0, 255)


def nudge(dist): return np.random.uniform(-dist, dist)


def project_on_grid(grid_size, result, base, baseMAP, colors=None):
    issues = 0
    grid = np.zeros(grid_size*grid_size)
    for index, value in enumerate(result):
        if grid[value] == 0 or grid[value] == base[index]:
            grid[value] = base[index]
        else:
            issues += 1
            print('Issue #'+str(issues)+'! Tried to change ' +
                  str(baseMAP[grid[value]])+' to ' + str(baseMAP[base[index]]))
    grid = grid.reshape(grid_size, grid_size)
    if colors == None:
        colormap = cm.get_cmap('seismic', len(np.unique(base.ravel())))
    else:
        colormap = mpl.colors.ListedColormap(colors)
    im = plt.imshow(grid, cmap=colormap, interpolation='none')

    # Get legends right
    # Solution found at https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib/39625380

    values = np.unique(base.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(
        l=baseMAP[values[i]])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def long_str_to_int_list(data):
    return np.array(list(filter(None, re.split(' |\n', data)))).astype('int')


def project_on_scatter(grid_size, result, base, baseMAP, colors=None):
    values = np.unique(base.ravel())
    if colors == None:
        np.random.seed(9)
        colors = ['#%02X%02X%02X' %
                  (r(), r(), r()) for i in range(len(np.unique(base.ravel()))+1)]
        patches = [mpatches.Patch(color=colors[i], label="District {l}".format(
            l=baseMAP[values[i]])) for i in range(len(values))]
    else:
        patches = [mpatches.Patch(color=colors[i], label="{l}".format(
            l=baseMAP[values[i]])) for i in range(len(values))]

    grid = -1*np.ones(grid_size*grid_size)
    for voter, winner_node in enumerate(result):
        winning_matrix_row = winner_node // grid_size
        winning_matrix_col = winner_node % grid_size
        if grid[winner_node] == -1:
            grid[winner_node] = base[voter]
            plt.scatter(winning_matrix_col, winning_matrix_row,
                        c=colors[base[voter]])
        else:
            plt.scatter(winning_matrix_col+nudge(0.5), winning_matrix_row+nudge(0.5),
                        c=colors[base[voter]], label=baseMAP[base[voter]])

    plt.legend(handles=patches, bbox_to_anchor=(
        0, 1), loc=2, borderaxespad=0.)
    plt.axis('equal')
    plt.show()


def plot(mode, on):
    n_inputs = votes.shape[1]
    n_data = votes.shape[0]
    n_nodes = 100
    repeat = True
    n_epochs = 10
    attempts = 1
    seed = False

    net = SOMNetwork(n_inputs=n_inputs,
                     n_nodes=n_nodes,
                     step_size=0.2,
                     topology='grid',
                     grid_shape=(10, 10),
                     neighbourhood_start=10,
                     neighbourhood_end=1,
                     n_epochs=n_epochs,
                     seed=seed)

    # result = np.random.randint(0, 100, n_data)

# 2o epochss seeded
    res1 = '''17  5 71 17 59 71 35 58 79  5  2 59 59 17 56  3 56  5 59 26 59 17 59 81
 59 15 59 15 79 28 15 19 56 59  4 17  4 17  8 26 59 59 59 59 23 59 71  3
 17 59 47 71  5 79 19 59 17  5  5 56 60  5  8 81 79  5 16 59 17 33 60 59
 17 59 59 14 19 59 15  5 17 39 59 59 99  3 59 79 46  5 60 71 59 45 59 59
 79  4 60 17 59  3 17  5 59 59 92  5 71 14  4 59 59 19 59 60 47  4 59 71
 5 17 60 37 59 19  3  5 60  2 92 17 35 59 59 59  5 59 59 59 59 81  3  5
 59 59 92 59 59 56 59 17  5 71 19  3 17 59 59 45 59 26  5 71 47 47  8 59
 5 18 59 17 17  3 17  5 57 60 17 13 59 46 59  5 59 60 17 39 59 59 59 59
 59  5 56  3 26 17 60 17 59 26 56 59  5 59 17 17 44  0 17 48  3 59 59 59
 5 15 59 35 17 17 37  2 60 59  5  8 26 59 59 59 67 59 17  8 59 33 26 59
 60 39 49  8 15  3 17 70 59 60  3 56 59 59 59  5 17 59 59 46 59 60 16 13
 17 77 59 70 59 59 92 71 46 79  5 60 59  2 59 59 49  5 17 59 47 59 60 46
 0 39  8 17 15 60 67 48 59 46 22 59 59 71 92 56 60 39 59  2 68 59  5 59
 59 59 47 59 60 45  3  5 17 59 59 99 17 46 60 59 19 92 45 15 58  3 71  8
 49  8 59 59 59 56 79  3 59 59 59 17  5'''


# 20 eps not seeded

    res2 = '''1  7  5  1 59  5 29 49 92  7 36 59 48  1 29 26 29  7 59 80 59  1 59  8
 59  7 59  7 29 19  7 91 29 59 28  1 28  1  8 80 59 59 59 59 81 59 23 26
 3 59 92  5  7 29 91 59  1  7  7 29  6  7  8  8 29  7 70 59  3 92  6 59
 1 59 59 27 91 59 70  7  1 39 59 59 99 26 59 29 29  7  6 23 59 29 59 59
 29 28  6 14 59 26  1  7 59 59 18  7 23 27 82 59 59 91 59  6 92 28 59 23
 7  1  6 91 59 91 26  7  6 36 18  1 29 48 59 59  7 59 59 59 59  8 26  7
 59 59 18 59 59 29 59  3  7 23 91 26  1 59 59 29 59 80  7 23 92 92  8 59
 7 91 59  1  1 26  1  7 29  6  1 27 59 29 59  7 59  6  1 39 59 59 59 59
 59  7 29 26 80  1  6  4 59 80 29 59  7 59  1  1 29 45  1 38 26 59 59 59
 7  7 59 29  4  4 19 36  6 59  7  8 80 59 59 59 29 59  1  8 59 92 80 59
 6 39 49  8  7 26  4 91 59  6 26 29 59 59 59  7  1 59 59 29 59  6  7 27
 1 29 59 91 59 59 18 23 29 29  7  6 59 36 59 59 49  7  1 59 92 59  6 29
 45 39  8  1  7  6 29 38 59 29 60 59 59 23 18 29  6 39 59 36 29 59  7 59
 59 59 92 59  6 29 26  7  1 59 59 99  1 29  6 59 91 18 29  7 67 26  5  8
 58  8 59 59 59 29 29 26 59 59 59  5  7'''

    res3_0_01leaky_20epochs = '''56 99 56 56  0 56 22  0  0 99 67  0  0 56 22 67 22 99  0 45  0 56  0 45
  0 89  0 89 11 34 89 33 22  0 45 56 45 56 44 45  0  0  0  0 24  0 56 67
 56  0  0 56 99  0 33  0 56 99 99 22 78 99 44 45  0 99 99  0 56 22 78  0
 56  0  0 67 33  0 89 99 56  0  0  0  0 67  0  0 22 99 78 56  0 22  0  0
  0 45 78 56  0 67 56 99  0  0 11 99 56 67 45  0  0 34  0 78  0 45  0 56
 99 56 78 33  0 33 67 89 78 67 11 56 22  0  1  0 99  0  0  0  0 45 67 99
  0  0 11  0  0 22  0 56 99 56 33 67 56  0  0 22  0 45 99 56  0  0 44  0
 99 44  0 56 56 67 56 99 23 78 56 67  0 22  0 99  0 78 56  0  0  0  0  0
  0 99 22 67 45 56 78 56  0 45 22  0 99  0 56 56 22 67 56 33 67  0  0  0
 99 99  0 22 56 56 33 67 78  0 99 44 45  0  0  0 23  0 56 44  0 22 45  0
 78  0  0 44 89 67 56 44  0 78 67 22  0  0  0 99 56  0  0 22  0 78 99 67
 56  0  0 34  0  0 11 56 22  0 99 78  0 67  0  0  0 99 56  0  0  0 78 22
 67  0 44 56 89 78 23 33  0 22 77  0  0 56 11 22 78  0  0 67  0  0 99  0
  0  0  0  0 78 22 67 99 56  0  0  0 56 22 78  0 33 11 22 89  0 67 55 44
  0 44  0  0  0 22  0 67  0  0  0 55 99'''

    res4_0_1leaky_10epochs = '''99 88 99 99  0 99 12  0  0 88 55  0  0 99 12 45 12 88  0 78  0 99  0 99
  0 88  0 88 11 77 88 67 12  0 45 99 45 99 77 78  0  0  0  0 13  0 99 45
 99  0  0 99 88  0 67  0 99 88 88 12 56 88 77 99  0 88 88  0 99 12 56  0
 99  0  0 45 67  0 88 88 99  0  0  0  0 45  0  0 12 88 56 99  0 12  0  0
  0 45 56 89  0 45 99 88  0  0  1 88 99 44 78  0  0 77  0 56  0 45  0 99
 88 99 56 77  0 67 45 88 56 55  1 99 12  0  1  0 88  0  0  0  0 99 45 88
  0  0  1  0  0 12  0 99 88 99 67 45 99  0  0 12  0 78 88 99  0  0 77  0
 88 77  0 99 99 45 99 88 22 55 99 45  0 12  0 88  0 56 99  0  0  0  0  0
  0 88 12 55 78 99 56 99  0 78 12  0 88  0 99 99 12 55 99  0 45  0  0  0
 88 88  0 12 99 99 77 55 55  0 88 77 78  0  0  0 22  0 99 77  0 12 78  0
 56  0  0 77 88 55 99 77  0 56 45 12  0  0  0 88 99  0  0 12  0 56 88 44
 99  0  0 67  0  0  1 99 12  0 88 56  0 55  0  0  0 88 99  0  0  0 56 12
 55  0 77 99 88 55 22  0  0 12 55  0  0 99  1 12 56  0  0 55  0  0 88  0
  0  0  0  0 56 12 45 88 99  0  0  0 99 12 56  0 67  1 12 88  0 45 99 77
  0 77  0  0  0 12  0 45  0  0  0 99 88'''

    res5_noleaky_10epochs_50neigh = '''0  7 14  0 69 14 44 99 29  7  8 69 79  0 36  8 36  7 69  7 69  0 69  5
 69  7 69  7 38  9  7  9 36 69  8  0  8  0  9  7 69 69 69 69 19 69  4  8
  1 69 29 14  7 49  9 69  0  7  7 36  8  7  9  5 49  7  7 69  2 33  8 69
  0 69 69  8  9 69  8  7  0 49 69 69 69  8 69 49 26  7  8  4 69 36 69 69
 49  8  8  4 69  8  0  7 69 69 48  7  4  8  8 69 69  9 69  8 29  8 69  4
  7  0  8 18 69  9  8  7  8  8 48  0 44 79 67 69  7 69 69 69 69  5  8  7
 69 69 48 69 69 36 69  2  7  4  9  8  0 69 69 36 69  7  7  4 29 29  9 69
  7  9 69  0  0  8  0  7 17  8  0  8 69 26 69  7 69  8  0 49 69 69 69 69
 69  7 36  8  7  0  8  4 69  7 36 69  7 69  0  0 26  8  0 29  8 69 69 69
  7  7 69 44  4  4 18  8  8 69  7  9  7 69 69 69 17 69  0  9 69 33  7 69
  8 49 69  9  7  8  4  9 69  8  8 37 69 69 69  7  0 69 69 26 69  8  7  8
  0 29 69  9 69 69 48  4 26 49  7  8 69  8 69 69 88  7  0 69 29 69  8 26
  8 49  9  0  7  8 17 29 69 37  8 69 69  4 59 37  8 49 69  8 29 69  7 69
 69 69 29 69  8 36  8  7  0 69 69 69  0 26  8 69  9 48 36  7 99  8 14  9
 68  9 69 69 69 36 49  8 69 69 69 12  7'''

    res5_noleaky_10epochs_100neigh = '''12  5 33 12 29 33 55 29 89  5  8 29 29 12 58  8 58  5 29 17 29 12 29  7
 29  5 29  5 79 49  5  9 58 29  8 12  8 12 18 17 29 29 29 29  9 29  4  8
 2 29 49 33  5 99  9 29 12  5  5 58  8  5 18  7 99  5 16 29 22 55  8 29
 12 29 29  8  9 29 26  5 12 29 29 29 29  8 29 99 59  5  8  4 29 67 29 29
 99  8  8 15 29  8 12  5 29 29 29  5  4  8  8 29 29  9 29  8 49  8 29  4
 5 12  8 49 29  9  8  5  8  8 29 12 47 29 29 29  5 29 29 29 29  7  8  5
 29 29 29 29 29 58 29 22  5  4  9  8 12 29 29 67 29 17  5  4 49 49 18 29
 5  9 29 12 12  8 12  5 69  6 12  8 29 59 29  5 29  8 12 29 29 29 29 29
 29  5 58  8 17 12  8  4 29 17 58 29  5 29 12 12 66  8 12 19  8 29 29 29
 5  5 29 47  4  0 49  8  6 29  5 18 17 29 29 29 69 29 12 18 29 55 17 29
 8 29 29 18  5  8  0 37 29  8  8 58 29 29 29  5 12 29 29 48 29  8  5  8
 12 69 29  9 29 29 29  4 59 99  5  8 29  8 29 29 29  5 12 29 49 29  8 59
 8 29 18 12  5  8 69 19 29 59  8 29 29  4 29 58  8 29 29  8 79 29  5 29
 29 29 49 29  8 67  8  5 12 29 29 29 12 59  8 29  9 29 67  5 29  8 34 18
 29 18 29 29 29 58 99  8 29 29 29  3  5'''

    res6_noleaky_10epochs_80neigh = '''2  0 12  2 99 12 57 99 78  0 33 99 99  2 66 33 66  0 99  3 99  2 99 45
 99  0 99  0 77 46  0 46 66 99 44  2 44  2 45  3 99 99 99 99 15 99 13 33
  2 99 68 12  0 78 55 99  2  0  0 66 24  0 45 45 78  0  0 99 13 56 24 99
  2 99 99 44 46 99 11  0  2 79 99 99 99 33 99 78 66  0 24 13 99 66 99 99
 78 44 24  2 99 33  2  0 99 99 78  0 13 34 35 99 99 46 99 24 68 44 99 13
  0  2 24 46 99 46 33  0 24 33 78  2 57 99 99 99  0 99 99 99 99 45 33  0
 99 99 78 99 99 66 99 13  0 13 46 33  2 99 99 66 99  3  0 13 68 68 45 99
  0 46 99  2  2 33  2  0 77 24  2 44 99 66 99  0 99 24  2 79 99 99 99 99
 99  0 66 33  3  2 24  2 99  3 66 99  0 99  2  2 57 33  2 79 33 99 99 99
  0  0 99 57  2  2 46 33 24 99  0 45  3 99 99 99 77 99  2 45 99 56  3 99
 24 79 99 45  0 33  2 45 99 24 33 57 99 99 99  0  2 99 99 66 99 24  0 34
  2 68 99 45 99 99 78 13 66 78  0 24 99 33 99 99 99  0  2 99 68 99 24 66
 33 79 45  2  0 24 77 79 99 57 44 99 99 13 79 57 24 79 99 33 68 99  0 99
 99 99 68 99 24 66 33  0  2 99 99 99  2 66 24 99 46 78 66  0 99 33 12 45
 99 45 99 99 99 66 78 33 99 99 99 22  0'''

    res7_invleaky_10epochs_80neigh = '''0  2  0  0 77  0 55 88 66  2 33 77 77  0 55 33 55  2 77  2 77  0 77 35
 77 12 77 12 56 44 12 44 55 77 34  0 34  0 35  2 77 77 77 77 44 77  0 33
  0 77 89  0 12 57 44 77  0  2  2 55 22  2 35 35 57  2  2 77  0 45 22 77
  0 77 77 24 44 77 11  2  0 79 77 77 68 24 77 57 55  2 22  0 77 55 77 77
 57 34 22  0 77 33  0  2 77 77 57  2  0 24 34 77 77 44 77 22 89 34 77  0
  2  0 22 99 77 44 33 12 22 33 57  0 55 77 66 77  2 77 77 77 77 35 33  2
 77 77 57 77 77 55 77  0  2  0 44 33  0 77 77 55 77  2  2  0 89 89 35 77
  2 35 77  0  0 33  0  2 56 22  0 24 77 55 77  2 77 22  0 79 77 77 77 77
 77  2 55 33  2  0 22  0 77  2 55 77  2 77  0  0 55 33  0 99 33 77 77 77
  2 12 77 55  0  0 99 33 22 77  2 35  2 77 77 77 56 77  0 35 77 45  2 77
 22 79 88 35 12 33  0 35 77 22 33 55 77 77 77  2  0 77 77 46 77 22  2 24
  0 57 77 44 77 77 57  0 55 57  2 22 77 33 77 77 79  2  0 77 89 77 22 55
 33 79 35  0 12 22 56 99 77 55 24 77 77  0 57 55 22 79 77 33 57 77  2 77
 77 77 89 77 22 55 33  2  0 77 77 68  0 55 22 77 44 57 55 12 68 33  0 35
 79 35 77 77 77 55 57 33 77 77 77  0  2'''

    res8_0_01leaky_10epochs_10neigh = '''67 99 78 67  0 78 23  0 12 99 56  0  0 67 23 56 23 99  0 79  0 67  0 68
  0 88  0 89 22 34 89 34 23  0 46 67 46 67 34 79  0  0  0  0 14  0 77 56
 67  0 11 78 89 12 33  0 67 99 99 23 57 99 34 68 12 99 99  0 67 33 57  0
 67  0  0 45 34  0 88 99 67  0  0  0  0 56  0 12 23 99 57 77  0 23  0  0
 12 46 57 77  0 56 67 99  0  0  2 99 77 55 44  0  0 34  0 57 11 46  0 77
 99 67 57  2  0 34 56 99 57 56  2 67 23  0  0  0 99  0  0  0  0 68 56 99
  0  0  2  0  0 23  0 67 99 77 34 56 67  0  0 23  0 79 99 77 11 11 34  0
 99 34  0 67 67 56 67 99 22 57 67 45  0 23  0 99  0 57 67  0  0  0  0  0
  0 99 23 56 79 67 57 78  0 79 23  0 99  0 67 67 23 56 67  2 56  0  0  0
 99 88  0 23 78 77 34 56 57  0 99 34 79  0  0  0 12  0 67 34  0 33 79  0
 57  0  0 34 89 56 77 34  0 57 56 23  0  0  0 99 67  0  0 23  0 57 99 55
 67 11  0 34  0  0  2 77 23 12 99 57  0 56  0  0  0 99 67  0 11  0 57 23
 56  0 34 67 89 57 12  2  0 23 45  0  0 77  2 23 57  0  0 56 12  0 99  0
  0  0 11  0 57 23 56 99 67  0  0  0 67 23 57  0 34  2 23 89  0 56 68 34
  1 34  0  0  0 23 12 56  0  0  0 67 99'''

    result = long_str_to_int_list(res8_0_01leaky_10epochs_10neigh)

    # result = net.fit(votes).astype('int')

    print(result)
    if on == 'grid':
        if mode == 'parties':
            project_on_grid(10, result, parties, PARTIES_MAP, COLORS_PARTIES)
        elif mode == 'sex':
            project_on_grid(10, result, sex, SEX_MAP, COLORS_SEX)
        elif mode == 'districts':
            _map = {i: i for i in range(np.max(districts)+1)}
            project_on_grid(10, result, districts, _map)
    elif on == 'scatter':
        if mode == 'parties':
            project_on_scatter(10, result, parties,
                               PARTIES_MAP, COLORS_PARTIES)
        elif mode == 'sex':
            project_on_scatter(10, result, sex, SEX_MAP, COLORS_SEX)
        elif mode == 'districts':
            _map = {i: i for i in range(np.max(districts)+1)}
            project_on_scatter(10, result, districts, _map)


plot('districts', 'scatter')
