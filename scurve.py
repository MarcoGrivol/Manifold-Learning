from sklearn import datasets, manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def fit_plot( x, y, n_neighbors, n_components ):
    isomap = manifold.Isomap( n_neighbors=n_neighbors, n_components=n_components )
    x_isomap = isomap.fit_transform( x )
    
    lle = manifold.LocallyLinearEmbedding( n_neighbors=10, n_components=2 )
    x_lle = lle.fit_transform( x )
    
    fig, (ax1, ax2) = plt.subplots( 1, 2 )
    
    x1_vals = x_isomap[:, 0]
    y1_vals = x_isomap[:, 1]
    ax1.scatter( x1_vals, y1_vals, c=y )
    
    x2_vals = x_lle[:, 0]
    y2_vals = x_lle[:, 1] 
    ax2.scatter( x2_vals, y2_vals, c=y )
    plt.show()

if __name__ == "__main__":
    x = np.loadtxt('data_scurve.txt')
    y = np.loadtxt('label_scurve.txt')

    idx = y.argsort()
    y.sort()
    x = x[idx]

    x = np.concatenate( (x[:270], x[300:]) )
    y = np.concatenate( (y[:270], y[300:]) )

    fit_plot( x, y, n_neighbors=10, n_components=2 )