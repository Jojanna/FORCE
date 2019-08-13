import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import random

# Make plot with horizontal colorbar
fig, ax = plt.subplots()

data = random((250,250)) + 3.5

norm = matplotlib.colors.Normalize(vmin=2.5,vmax=4.5)

cax = ax.imshow(data, interpolation='nearest', cmap=cm.afmhot, norm=norm)
ax.set_title('Gaussian noise with horizontal colorbar')

TICKS = [1,2,3,4]

cbar = fig.colorbar(cax, ticks=TICKS, orientation='horizontal')

# the following command extracts the first tick object from the x-axis of
# the colorbar:
tick = cbar.ax.get_xaxis().get_major_ticks()[0]

# Here you compare the text of the first tick label to all the tick locations
# you have defined in TICKS (they need to be strings for this):
CUSTOM_INDEX = [str(S) for S in TICKS].index(tick.label1.get_text())

TICKLABELS = ['one','two', 'three', 'four']

# Now, you can use the index of the actual first tick as a starting tick to
# your list of custom labels:
cbar.set_ticklabels(TICKLABELS[CUSTOM_INDEX:])# horizontal colorbar

plt.show()