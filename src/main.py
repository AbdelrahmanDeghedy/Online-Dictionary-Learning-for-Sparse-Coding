from odl import OnlineDictionaryLearning
from DataGen import generate
from plot import plot_many

# Data parameters
frequenciesList = [0.02 * i for i in range(15)]
numberOfObservations = 10000
timeInterval = [0, 100]
segmentSize = 100
std_deviation = 0.1
coefficients_range = [-2, 2]
sparsity = 0.8

# Learning parameters
iterations = 1000
dictionary_size = 10
regularizationParameter = 0.001 # lambda

# Generate data
data = generate(frequenciesList, numberOfObservations, timeInterval, segmentSize, std_deviation, coefficients_range, sparsity)

# Visualize data
# x = [i for i in range(segmentSize)]
# plot_many(x, computedDictionary, x_label="x", y_label="predicted_dictionary", filename="predicted_dictionary-lambda-" + str(regularizationParameter) + str(std_deviation), to_save=True)

# Learn the dictionaries
model = OnlineDictionaryLearning(data)
model.learn(iterations, regularizationParameter, dictionary_size)


x = [i for i in range(len(model.cumulative_losses))]
plot_many(x, [model.cumulative_losses], x_label="Iterations", y_label="Cumulative losses", filename = "cumulative_losses-lambda-" + str(regularizationParameter), to_save=True)
