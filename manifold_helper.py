from sklearn import datasets, manifold, mixture, metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats, linalg
import itertools
from math import ceil

def fit_plot( x, n_neighbors, n_components, y_colors ):
    isomap = manifold.Isomap( n_neighbors=n_neighbors, n_components=n_components )
    y_isomap = isomap.fit_transform( x )
    
    lle = manifold.LocallyLinearEmbedding( n_neighbors=n_neighbors, n_components=n_components )
    y_lle = lle.fit_transform( x )
    
    laplace = manifold.SpectralEmbedding( n_neighbors=n_neighbors, n_components=n_components )
    y_laplace = laplace.fit_transform( x )
    
    ltsa = manifold.LocallyLinearEmbedding( n_neighbors=n_neighbors, n_components=n_components, method='ltsa' )
    y_ltsa = ltsa.fit_transform( x )
    
    fig = plt.figure()
    gs = fig.add_gridspec( 2, 2 )
    ax1 = fig.add_subplot( gs[0, 0] )
    ax2 = fig.add_subplot( gs[0, 1] )
    ax3 = fig.add_subplot( gs[1, 0] )
    ax4 = fig.add_subplot( gs[1, 1] )
    
    ax1.set_title( 'isomap' )
    ax2.set_title( 'lle' )
    ax3.set_title( 'laplace' )
    ax4.set_title( 'ltsa' )
    
    x1_vals = y_isomap[:, 0]
    y1_vals = y_isomap[:, 1]
    ax1.scatter( x1_vals, y1_vals, c=y_colors )
    ax1.set_xticklabels( [] )
    ax1.set_yticklabels( [] )
    
    x2_vals = y_lle[:, 0]
    y2_vals = y_lle[:, 1] 
    ax2.scatter( x2_vals, y2_vals, c=y_colors )
    ax2.set_xticklabels( [] )
    ax2.set_yticklabels( [] )
    
    x3_vals = y_laplace[:, 0]
    y3_vals = y_laplace[:, 1]
    ax3.scatter( x3_vals, y3_vals, c=y_colors )
    ax3.set_xticklabels( [] )
    ax3.set_yticklabels( [] )
    
    x4_vals = y_ltsa[:, 0]
    y4_vals = y_ltsa[:, 1]
    ax4.scatter( x4_vals, y4_vals, c=y_colors )
    ax4.set_xticklabels( [] )
    ax4.set_yticklabels( [] )
    
    plt.show()
    
    return (y_isomap, y_lle, y_laplace, y_ltsa)

def gmm_results(X, Y_, means, covariances, title, colors, index=0):
#     color_iter = itertools.cycle(['purple', 'blue'])
    color_iter = itertools.cycle( colors )
    fig = plt.figure()
    splot = fig.add_subplot()
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()
    
def get_true_labels( y ):
    true_labels = []
    for c in y:
        if c == 'purple':
            true_labels.append( 0 )
        else:
            true_labels.append( 1 )
    true_labels.reverse()
    return true_labels

def isomap( data, n_neighbors, n_components ):
    return manifold.Isomap( n_neighbors=n_neighbors, n_components=n_components ).fit_transform( data )

def GMM( data, labels, n_components ):
    gmm = mixture.GaussianMixture( n_components=n_components, covariance_type='full' )
    gmm = gmm.fit( data )
    return metrics.adjusted_rand_score( labels, gmm.predict( data ) )
    
def lle( data, n_neighbors, n_components ):
    return manifold.LocallyLinearEmbedding( n_neighbors=n_neighbors, n_components=n_components ).fit_transform( data )

def laplace( data, n_neighbors, n_components ):
    return manifold.SpectralEmbedding( n_neighbors=n_neighbors, n_components=n_components ).fit_transform( data )

def ltsa( data, n_neighbors, n_components ):
    return manifold.LocallyLinearEmbedding( n_neighbors=n_neighbors, n_components=n_components, method='ltsa' ).fit_transform( data )

def ari_results( data, n_components, step, rnge, labels, colors ):
    y_isomap = []
    y_lle = []
    y_laplace = []
    y_ltsa = []
    neighbors_range = [i for i in rnge if i % step == 0]
    
    erro = False
    
    for i in neighbors_range:
        y_isomap.append( isomap( data, i, n_components ) )
        y_lle.append( lle( data, i , n_components ) )
        y_laplace.append( laplace( data, i, n_components ) )
        try:
            y_ltsa.append( ltsa( data, i, n_components ) )
        except:
            erro = True
        
    ari_isomap = []
    ari_lle = []
    ari_laplace = []
    ari_ltsa = []
    for i in range( len( y_isomap ) ):
        ari_isomap.append( GMM( y_isomap[i], labels, n_components ) )
        ari_lle.append( GMM( y_lle[i], labels, n_components ) )
        ari_laplace.append( GMM( y_laplace[i], labels, n_components ) )
        if not erro:
            ari_ltsa.append( GMM( y_ltsa[i], labels, n_components ) )
            
    fig = plt.figure()
    ax = fig.add_subplot()
    
    x_vals = neighbors_range
    ax.scatter( x_vals, ari_isomap, c=colors[0], label="Isomap" )
    ax.scatter( x_vals, ari_lle, c=colors[1], label="LLE" )
    ax.scatter( x_vals, ari_laplace, c=colors[2], label="Laplace" )
    if not erro:
        ax.scatter( x_vals, ari_ltsa, c=colors[3], label="LTSA" )
    ax.legend()