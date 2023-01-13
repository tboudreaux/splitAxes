from splitAxes import multiAxes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def split_grid(figRows, figColums, splitMatrix, figsize=(10,7)):
    gridNcols = np.lcm.reduce((splitMatrix.ravel()+1)*figColums)
    splitsPerRow = gridNcols // figColums

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = gridspec.GridSpec(ncols=gridNcols,nrows=figRows, figure=fig)

    axes = np.empty(shape=(figRows, gridNcols), dtype='object')
    figureFractionalWidth = 1 / figColums
    for rowID, rowSplits in enumerate(splitMatrix):
        for colID, colSplits in enumerate(rowSplits):
            gridPointsPerSubFigure = int((gridNcols * figureFractionalWidth)*(1/(colSplits+1)))
            figureOffset = int(gridNcols * figureFractionalWidth * colID)
            splitAxes = multiAxes(disableYticks=True if colID > 0 else False)
            for splitID in range(colSplits+1):
                subFigureOffset = splitID * gridPointsPerSubFigure
                ax = fig.add_subplot(grid[rowID,subFigureOffset+figureOffset:figureOffset+subFigureOffset+gridPointsPerSubFigure])
                splitAxes.append(ax)
            splitAxes.commit()
            axes[rowID,colID] = splitAxes
    return fig, axes
