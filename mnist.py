from sklearn import (datasets, manifold, random_projection)
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_embedding

digits = datasets.load_digits()
x = digits.data
y = digits.target
n_samples, n_features = x.shape
n_neighbors = 30

# plot image of the digits
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = x[i * n_img_per_row + j].reshape((8, 8))

# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')
# plt.show()
# ---------------------------------------

# random 2d projection using a random unitary matrix
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# x_projected = rp.fit_transform(x)
# plot_embedding(dataset=digits, X=x_projected, targets=y)
# plt.show()
# ---------------------------------------

# ISOMAP
# x_isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(x)
# plot_embedding(dataset=digits, X=x_isomap, targets=y)
# plt.show()
# ---------------------------------------

# LLE
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='standard')
x_lle = clf.fit_transform(x)
plot_embedding(dataset=digits, X=x_lle, targets=y)
plt.show()
# ---------------------------------------