import numpy as np

X = np.array(([3,5],[5,1],[10,2]))
y = np.array((75,82,93))

X= X/np.amax(X,axis=0)
y = y/100