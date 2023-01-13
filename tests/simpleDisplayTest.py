from splitAxes import split_grid
import numpy as np
import matplotlib.pyplot as plt

figRows = 2
figColums = 2
splitMatrix=np.array([[3,1],[0,2]])

fig, axs = split_grid(figRows, figColums, splitMatrix, figsize=(15,7))

# Example Data
X = np.linspace(-10,50, 1000)
Y = np.exp(-(X-0)**2) + np.exp(-(X-10)**2) + np.exp(-(X-20)**2)

axs[0,0].set_xlim(0,45)
axs[0,0].set_ylabel("Bob")
axs[1,0].plot(X,Y, color='green')
axs[1,1].scatter([0,3,4],[5,4,2])
axs[0,1].set_xlabel("Dave", position="manual", labelpos=0.67)

axs[1,0].fill_between([0,10], 0.3, alpha=0.5, color='blue')

plt.show()
# print(axs[0,0].get_xlim())
