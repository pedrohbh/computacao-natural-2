import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pyswarms as ps

data = pd.read_csv('waveform.csv')
for c in data.columns:
    if c not in ['classe']:
        c_max = max(data[c])
        c_min = min(data[c])
        data[c] = (data[c] - c_min) / (c_max - c_min)

#imprime as primeiras linhas
print('head():'); print(data.head())
print('-----------------------------------------------')
#imprime as últimas linhas
print('tail():'); print(data.tail())
print('-----------------------------------------------')

X = data.drop(columns=['classe'])
y = data[['classe']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#imprime as primeiras linhas
print('head():'); print(X.head())
print('-----------------------------------------------')
#imprime as últimas linhas
print('tail():'); print(X.tail())
print('-----------------------------------------------')

#imprime as primeiras linhas
print('head():'); print(x_train.head())
print('-----------------------------------------------')
#imprime as últimas linhas
print('tail():'); print(x_train.tail())
print('-----------------------------------------------')

n_camada_de_entrada = 21
n_camada_oculta = 20
n_camada_saida = 3

#numero_dados = 5000

def logits_function(p, X_selecionado):
    W1 = p[0:420].reshape((n_camada_de_entrada, n_camada_oculta))
    b1 = p[420:440].reshape((n_camada_oculta,))
    W2 = p[440:500].reshape((n_camada_oculta,n_camada_saida))
    b2 = p[500:503].reshape((n_camada_saida,))

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
otimizador = ps.single.GlobalBestPSO(n_particles=25, dimensions=dimensoes, options=opcoes)

cost, pos = otimizador.optimize(f, iters=100, X_selecionado=x_train.to_numpy(), Y_selecionado=y_train.to_numpy())

def predict(pos, X_selecionado):
    logits = logits_function(pos, X_selecionado)
    y_pred = np.argmax(logits, axis=1)
    return y_pred

print("Acurácia encontrada: {}".format((predict(pos, x_test.to_numpy()) == y_test.to_numpy()).mean()))
