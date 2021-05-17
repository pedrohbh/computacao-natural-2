from operator import mod
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf

data = load_iris()

X = data.data
y = data.target

# Processo de normalização
X = preprocessing.normalize(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



def fitness_func(solution, sol_idx):
  global data_inputs, data_outputs, keras_ga, model

  model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)

  model.set_weights(weights=model_weights_matrix)

  predictions = model.predict(data_inputs)

  mae = tensorflow.keras.losses.MeanAbsoluteError()
  abs_error = mae(data_outputs, predictions).numpy() + 0.00000001
  solution_fitness = 1.0 / abs_error

  return solution_fitness



def callback_generation(ga_instance):
  print("Generation = {generation}".format(generation=ga_instance.generations_completed))
  print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


model = tf.keras.Sequential([
  tf.keras.layers.Dense(20, activation=tf.nn.tanh, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(3)
])

#weights_vector = pygad.kerasga.model_weights_as_vector(model=model)

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

data_inputs = x_train

data_outputs = tensorflow.keras.utils.to_categorical(y_train)

num_generations = 200
num_parents_mating = 5
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func)

ga_instance.run()

ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
# print("Predictions : \n", predictions)

# Calculate the categorical crossentropy for the trained model.
mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error)

# Calculate the classification accuracy for the trained model.
ca = tensorflow.keras.metrics.CategoricalAccuracy()
ca.update_state(data_outputs, predictions)
accuracy = ca.result().numpy()
print("Accuracy : ", accuracy)