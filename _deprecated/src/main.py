from odl import OnlineDictionaryLearning
from DataGen import generate
from plot import plot_many

# Setting the Data parameters
frequenciesList = [0.02 * i for i in range(15)]
numberOfObservations = 10000
timeInterval = [0, 250]
segmentSize = 100
std_deviation = 0.1
coefficients_range = [-2, 2]
sparsity = 0.8

# Setting the Learning parameters
iterations = 10000
dictionary_size = 10
regularizationParameter = 0.001 # lambda

# Generate data
data = generate(frequenciesList, numberOfObservations, timeInterval, segmentSize, std_deviation, coefficients_range, sparsity)

# Learn the dictionaries
model = OnlineDictionaryLearning(data)
model.learn(iterations, regularizationParameter, dictionary_size)

# Plotting the Objective Function VS Time
xAxis = [i for i in range(timeInterval[0], timeInterval[1])]
title = "objective_function_with_time, using lambda = " + str(regularizationParameter)
plot_many(xAxis, [model.objective], x_label="Time", y_label="Objective Function", title = title, filename = title, to_save=True)
