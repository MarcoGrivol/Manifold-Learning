from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import numpy as np

digits = load_digits(return_X_y=True)
X = digits[0]
target = digits[1]

neighbors = 5
nbrs = NearestNeighbors(n_neighbors=neighbors)
nbrs.fit(X)
G = nbrs.kneighbors_graph(X)
G = G.toarray()
