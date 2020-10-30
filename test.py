import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np

def lerp( a, b, t ):
    return (1 - t) * a + t * b

x, y = make_blobs( n_samples=10, centers=3, n_features=3)
x, y, x[:, 0]

fig = plt.figure()
ax = fig.add_subplot( 111, projection='3d' )

x_vals = x[:, 0]
y_vals = x[:, 1]
z_vals = x[:, 2]

ax.scatter( x_vals, y_vals, z_vals )
plt.show()

df = pd.DataFrame( x )
smm = np.array( squareform( pdist( df ) ) )
smm

a = np.min( smm[np.nonzero(smm)] )
b = np.max( smm[np.nonzero(smm)] )
for i in range( len( smm ) ):
    for j in range( len( smm ) ):\
        if i == 0 and j == 0:
            smm[i, j] = 0
        else:
            v = smm[i, j]
            smm[i, j] = lerp( a, b, v/b )
smm