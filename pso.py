import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets import load_iris

import pyswarms as ps

data = load_iris()

X = data.data
Y = data.target

n_camada_de_entrada = 4
n_camada_oculta = 20
n_camada_saida = 3

numero_dados = 150

def logits_function(p):
    W1 = p[0:80].reshape((n_camada_de_entrada, n_camada_oculta))
    b1 = p[80:100].reshape((n_camada_oculta,))
    W2 = p[100:160].reshape((n_camada_oculta,n_camada_saida))
    b2 = p[160:163].reshape((n_camada_saida,))

    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1) # Ativação da primeira camada
    logits = a1.dot(W2) + b2
    return logits