from sklearn import manifold
from collections import OrderedDict

class ManifoldHelper:

    def __init__(self, n_components, n_neighbors):
        self.transformed = OrderedDict()
        self.methods = OrderedDict()
        self.methods['ISOMAP'] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
        self.methods['LLE'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
        self.methods['SE'] = manifold.SpectralEmbedding(n_neighbors=n_neighbors, n_components=n_components)
        self.methods['LTSA'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='ltsa')

    def fitTransform(self, X, label):
        self.transformed[label] = self.methods[label].fit_transform(X)
