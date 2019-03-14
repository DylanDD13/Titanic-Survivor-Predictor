import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def normalize(X):
    return sc.fit_transform(X)
    