from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path

import matplotlib.pyplot as plt
import numpy as np

digits = load_digits(return_X_y=True)
X = digits[0]
target = digits[1]

# KNN Graph
neighbors = 5
knn = NearestNeighbors(n_neighbors=neighbors)
knn.fit(X)
knnGraph = knn.kneighbors_graph(X)

# Pairwise distance matrix nxn
D = graph_shortest_path(knnGraph, directed=False)

# MDS
A = -(1/2) * D
n = A.shape[0]

I = np.identity(n)
U = np.ones_like(A)
H = I - ((1/n) * U)

B = H*A*H

Y = np.linalg.eig(B)
w = Y[0]
v = Y[1]

m = [0, 0]
for i in range(w.shape[0]):
    if w[i] > w[m[0]]:
        m[0] = i
    elif w[i] > w[m[1]]:
        m[1] = i


