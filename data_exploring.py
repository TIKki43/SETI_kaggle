from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib
from pylab import *

arr = np.load('/home/timur/Documents/Projects/SETI/train/0/001c619bdf53.npy').astype(float)
fig = plt.figure()
for i in range(6):
    ax = fig.add_subplot(3, 3, i+1)
    imgplot = plt.imshow(arr[i])
plt.show()

cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.5, 1.0, 0.7),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.0),
                  (1.0, 0.5, 1.0))}

fig = plt.figure()
for i in range(6):
    my_cmap = matplotlib.colors.LinearSegmentedColormap('', cdict,256)
    ax = fig.add_subplot(3, 3, i + 1)
    imgplot = (plt.pcolor(arr[i], cmap=my_cmap))
plt.show()

import matplotlib.colors as mcolors

for i in range(6):
    data = arr[i]
    norm = mcolors.LogNorm(arr[i].mean() + 0.5 * arr[i].std(), arr[i].max())
    gammas = [0.8, 0.5, 0.3]
    fig, axs = plt.subplots(nrows=2, ncols=2)

    for ax, gamma in zip(axs.flat[1:], gammas):
        imgplot = ax.hist2d(data[:, 0], data[:, 1], norm=mcolors.PowerNorm(gamma))

    fig.tight_layout()

    plt.show()