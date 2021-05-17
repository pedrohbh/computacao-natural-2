# evolution strategy (mu, lambda) of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.core.records import array
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    j = forward_propagation(x, X_selecionado, Y_selecionado)
    return j


# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# evolution strategy (mu, lambda) algorithm
def es_comma(objective, bounds, n_iter, step_size, mu, lam, x_dados, y_dados):
	best, best_eval = None, 1e+10
	# calculate the number of children per parent
	n_children = int(lam / mu)
	# initial population
	population = list()
	for _ in range(lam):
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		population.append(candidate)
	# perform the search
	for epoch in range(n_iter):
		# evaluate fitness for the population
		scores = [objective(c, x_dados, y_dados) for c in population]
		# rank scores in ascending order
		ranks = argsort(argsort(scores))
		# select the indexes for the top mu ranked solutions
		selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
		children = list()
		for i in selected:
			# check if this parent is the best solution ever seen
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
			# create children for parent
			for _ in range(n_children):
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		# replace population with children
		population = children
	return [best, best_eval]


# seed the pseudorandom number generator
seed(1)
# define range for input

dimensoes = (n_camada_de_entrada * n_camada_oculta) + (n_camada_oculta * n_camada_saida) + n_camada_oculta + n_camada_saida
n_elementos = dimensoes
vetor_de_limites = []
lower_bound = -20
upper_bound = 20

for _ in range(n_elementos):
    vetor_de_limites.append([lower_bound, upper_bound])
bounds = asarray(vetor_de_limites)
# define the total iterations
n_iter = 5000
# define the maximum step size
step_size = 0.15
# number of parents selected
mu = 20
# the number of children generated by parents
lam = 100
# perform the evolution strategy (mu, lambda) search
best, score = es_comma(f, bounds, n_iter, step_size, mu, lam, x_train, y_train)
print('Done!')
print('f(%s) = %f' % (best, score))

def predict(pos, X_selecionado):
    logits = logits_function(pos, X_selecionado)
    y_pred = np.argmax(logits, axis=1)
    return y_pred

print("Acurácia encontrada: {}".format((predict(best, x_test) == y_test).mean()))



"""
otimizador = ps.single.GlobalBestPSO(n_particles=25, dimensions=dimensoes, options=opcoes)

cost, pos = otimizador.optimize(f, iters=1000, X_selecionado=x_train, Y_selecionado=y_train)

def predict(pos, X_selecionado):
    logits = logits_function(pos, X_selecionado)
    y_pred = np.argmax(logits, axis=1)
    return y_pred

print("Acurácia encontrada: {}".format((predict(pos, x_test) == y_test).mean()))
"""