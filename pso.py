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


def forward_propagation(params):
    logits = logits_function(params)

    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    corect_logprobs = -np.log(probs[range(numero_dados), y])
    loss = np.sum(corect_logprobs) / numero_dados

    return loss

def f(x):
    num_particulas = x.shape[0]
    j = [forward_propagation(x[i]) for i in range(numero_particulas)]
    return np.array(j)

# Aplicação do PSO
