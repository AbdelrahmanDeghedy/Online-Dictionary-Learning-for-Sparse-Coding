import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import plotMatrixHeatMap, running_average, sample_columns
# Provides a progress bar
from tqdm import tqdm



# Defining the seed
np.random.seed(1)

# Defining the parameters
lambda_value = 0.1
num_iterations = 100

# Defining the dimensions
m = 10
n = 100
k = 20

# Dimensions: m x n
X = np.random.randn(m, n)


def initializeD (m, k) :
    # random_matrix = X[:, np.random.choice(n, k, replace=False)]

    # # Generate a random matrix
    random_matrix = np.random.rand(m, k)

    # Replace some elements in each column with zeros
    zero_indices = np.random.choice(k, size=int(k/2), replace=False)
    random_matrix[:, zero_indices] = 0

    # Normalize the columns
    D_init = random_matrix / max([1, np.linalg.norm(random_matrix)])

    return D_init 

def update_dictionary(A, B, dictionary, dict_cols):
    # For each column of the dictionary
    for j in range(dict_cols) :
        u_j = dictionary[:, j] + (B[:, j] - np.matmul(dictionary, A[:, j])) / A[j, j]
        dictionary[:, j] = u_j / max([1, np.linalg.norm(u_j)])
    
    return dictionary

def learn(samples, num_iterations, plotMatrixDifference = False):
    t0 = 0.1
    objective_values = []
    # Initialize the accumulated objective function
    accumulatedObjective = 6
    
    D = initializeD(m, k)
    # print(D)

    # From section 3.4.4 Slowing Down The First Iterations Initializations
    A = t0 * np.eye(k, k)
    B = t0 * D

    for i in tqdm(range(num_iterations)):
        # Draw x t from p(x).
        X_t = sample_columns(X, samples)

        # Define the optimization variables
        currAlpha = cp.Variable((k, samples))

        # Define the objective function
        objective = cp.Minimize((1 / 2) * cp.sum_squares(X_t - np.matmul(D, currAlpha)) + lambda_value * cp.norm(currAlpha, 1))


        constraints = []
        for j in range(k):
            constraints += [
                cp.norm(D[:, j], 2) <= 1,
            ]

        # Define the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        optimalObjectiveValue = problem.solve()

        # Retrieve the optimized alpha
        optimized_alpha = currAlpha.value

        A += (1/2) * (np.dot(optimized_alpha, optimized_alpha.T) / samples)
        B += np.matmul(X_t, optimized_alpha.T) / samples

        D = update_dictionary(A, B, D, k)    
        
        accumulatedObjective = running_average(optimalObjectiveValue, accumulatedObjective, i + 1)
        objective_values.append(accumulatedObjective)

        if (i == 0 or i == num_iterations - 1) and plotMatrixDifference and samples == 1:
            x_hat = np.around(np.matmul(D, optimized_alpha), decimals=1)
            diff_matrix = np.abs(X - x_hat)
            plotMatrixHeatMap(diff_matrix, f"Matrix_Reconstruction_Difference_(Iteration{i + 1}, with K = {k})")

        if samples == 1 and i == num_iterations - 1:
            x_hat = np.around(np.matmul(D, optimized_alpha), decimals=1)
            plotMatrixHeatMap(x_hat, f"Matrix_Reconstruction_(Iteration{i + 1}, with K = {k})")
            plotMatrixHeatMap(D, f"Dictionary_(Iteration{i + 1}, with K = {k}), with D initialized randomly")
            plotMatrixHeatMap(optimized_alpha.T, f"Alpha_(Iteration{i + 1}, with K = {k})")

        if samples == 1 and i == num_iterations - 1:
            plotMatrixHeatMap(optimized_alpha.T, f"Alpha_(Iteration{i + 1}, with K = {k})")
        
    return objective_values


# Defining the patch sizes
patchSizes = [1, 4, 10]
labels = ["Our Method", f"Batch n = {patchSizes[1]}", f"Batch n = {patchSizes[2]}"]
times = [i for i in range(1, num_iterations + 1)]

# List of lists for all objective values in different patch sizes

objective_values = []

# Learning and Plotting the Objective Function VS Time
for i, samples in enumerate(patchSizes):
    objective_values.append(learn(samples, num_iterations, plotMatrixDifference = True))

for i, currObjValues in enumerate(objective_values):
    plt.plot(times, currObjValues, label=labels[i])

# Showing the plot
title = 'Objective function vs. time, using random columns of the dataset as initialization for D'
plt.title(title)
plt.xlabel('Time')
plt.ylabel('Objective function')
plt.legend()
plt.savefig(f'../results/{"_".join(title.split())}.png', bbox_inches='tight')
plt.show()
