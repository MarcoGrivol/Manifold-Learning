import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

n_points = 1000
points = datasets.make_s_curve( n_points, random_state=0)
X = points[0]
color = points[1]
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))

# add 3d scatter plot
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

# setup manifold methods
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
methods = OrderedDict()
methods['Isomap_10'] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
methods['LLE_10'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
methods['Isomap_50'] = manifold.Isomap(n_neighbors=n_neighbors+40, n_components=n_components)
methods['LLE_50'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors+40, n_components=n_components, method='standard')
methods['Isomap_75'] = manifold.Isomap(n_neighbors=n_neighbors+65, n_components=n_components)
methods['LLE_75'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors+65, n_components=n_components, method='standard')
methods['Isomap_100'] = manifold.Isomap(n_neighbors=n_neighbors+90, n_components=n_components)
methods['LLE_100'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors+90, n_components=n_components, method='standard')


# plot
for i, (label, method) in enumerate(methods.items()):
    Y = method.fit_transform(X)
    ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("%s" % (label))

plt.show()