import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from som_net import SOMNetwork
import re
# np.set_printoptions(threshold=sys.maxsize)
# plt.style.use('ggplot')
# plt.grid(False)
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
COLORS_PARTIES = ['#333333', '#4363bd', '#911eb4',
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
        np.random.seed(1)
        colors = ['#%02X%02X%02X' %
                  (r(), r(), r()) for i in range(np.unique(base.ravel()))]
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


def plot(mode):
    n_inputs = votes.shape[1]
    n_data = votes.shape[0]
    n_nodes = 100
    repeat = True
    n_epochs = 1
    attempts = 1
    seed = False

    net = SOMNetwork(n_inputs=n_inputs,
                     n_nodes=n_nodes,
                     step_size=0.1,
                     topology='grid',
                     grid_shape=(10, 10),
                     neighbourhood_start=4,
                     neighbourhood_end=1,
                     n_epochs=n_epochs,
                     seed=seed)
    fit = net.fit(votes)
    result = np.random.randint(0, 100, n_data)

    if mode == 'parties':
        project_on_grid(10, result, parties, PARTIES_MAP, COLORS_PARTIES)
    elif mode == 'sex':
        project_on_grid(10, result, sex, SEX_MAP, COLORS_SEX)
    elif mode == 'districts':
        _map = {i: i for i in range(np.max(districts)+1)}
        project_on_grid(10, result, districts, _map)
        # COLORS_DISTRICTS = ['#%02X%02X%02X' %
        #                     (r(), r(), r()) for i in range(10)]
        # print(COLORS_DISTRICTS)
        # project_on_grid(10, result, sex, SEX_MAP, COLORS_SEX)


plot('parties')
