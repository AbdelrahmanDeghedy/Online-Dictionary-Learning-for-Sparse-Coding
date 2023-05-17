import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
# Provides a progress bar
from tqdm import tqdm

# Defining the seed
np.random.seed(0)

# Defining the parameters
lambda_value = 0.1
num_iterations = 100

# Defining the dimensions
m = 20
n = 100
k = 20

# Defining the data set
# Dimensions: m x n
# X = np.array([
#                 [1, 3, 5, 2],
#                 [2, 4, 1, 3]
#             ]) 


X = np.random.randn(m, n)
# Dimensions: m x k
D = np.random.randn(m, k)
D /= np.linalg.norm(D, axis=0)

# D = np.array([
#                 [1 / math.sqrt(5), 5 / math.sqrt(26)],
#                 [2 / math.sqrt(5), 1 / math.sqrt(26)]
#             ])

A = np.random.randn(k, k)
B = np.random.randn(m, k)


def update_dictionary(a, b, dictionary, dict_cols):
    # For each column of the dictionary
    for j in range(dict_cols) :
        u_j = dictionary[:, j] + (b[:, j] - np.matmul(dictionary, a[:, j])) / a[j, j]
        dictionary[:, j] = u_j / max([1, np.linalg.norm(u_j)])

    return dictionary


objective_values = []
times = [i for i in range(1, num_iterations + 1)]

for i in tqdm(range(num_iterations)):
    # Define the optimization variables
    # Dimensions: k x n
    currAlpha = cp.Variable((k, n))

    # Define the objective function
    objective = cp.Minimize(cp.sum_squares(X - np.matmul(D, currAlpha)) + lambda_value * cp.norm(currAlpha, 1))

    constraints = []
    # Columns of D are normalized
    for j in range(k):
        constraints += [
            cp.norm(D[:, j]) <= 1,
        ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Retrieve the optimized alpha
    optimized_alpha = currAlpha.value

    # get the optimal objective value
    objective_values.append(problem.value)

    A += (1/2) * np.matmul(optimized_alpha, optimized_alpha.T)
    B += (1/2) * np.matmul(X, optimized_alpha.T)

    update_dictionary(A, B, D, k)

    x_hat = np.matmul(D, optimized_alpha)
    # print("Reconstructed:\n", D)


#  Plot the objective function with time
plt.plot(times[1:], objective_values[1:])
plt.xlabel('Time')
plt.ylabel('Objective function')
plt.show()
