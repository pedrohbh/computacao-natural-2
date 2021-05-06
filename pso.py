import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pyswarms as ps

data = load_iris()

X = data.data
y = data.target

# Processo de normalização
X = preprocessing.normalize(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

n_camada_de_entrada = 4
n_camada_oculta = 20
n_camada_saida = 3

numero_dados = 150

def logits_function(p, X_selecionado):
    W1 = p[0:80].reshape((n_camada_de_entrada, n_camada_oculta))
    b1 = p[80:100].reshape((n_camada_oculta,))
    W2 = p[100:160].reshape((n_camada_oculta,n_camada_saida))
    b2 = p[160:163].reshape((n_camada_saida,))

    z1 = X_selecionado.dot(W1) + b1
    a1 = np.tanh(z1) # Ativação da primeira camada
    logits = a1.dot(W2) + b2
    return logits


def forward_propagation(params, X_selecionado, Y_selecionado):
    logits = logits_function(params, X_selecionado)
    numero_dados = Y_selecionado.shape[0]

    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    corect_logprobs = -np.log(probs[range(numero_dados), Y_selecionado])
    loss = np.sum(corect_logprobs) / numero_dados

    return loss

def f(x, X_selecionado, Y_selecionado):
    num_particulas = x.shape[0]
    j = [forward_propagation(x[i], X_selecionado, Y_selecionado) for i in range(num_particulas)]
    return np.array(j)

# Aplicação do PSO
opcoes = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}


dimensoes = (n_camada_de_entrada * n_camada_oculta) + (n_camada_oculta * n_camada_saida) + n_camada_oculta + n_camada_saida
otimizador = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensoes, options=opcoes)

cost, pos = otimizador.optimize(f, iters=1000, X_selecionado=x_train, Y_selecionado=y_train)

def predict(pos, X_selecionado):
    logits = logits_function(pos, X_selecionado)
    y_pred = np.argmax(logits, axis=1)
    return y_pred

print("Acurácia encontrada: {}".format((predict(pos, x_test) == y_test).mean()))