import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets import load_iris

import pyswarms as ps

data = load_iris()

X = data.data
Y = data.target