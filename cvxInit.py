import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
# Provides a progress bar
from tqdm import tqdm

def plotDifferenceMatrix(matrix1, matrix2, fileSaveName = "absolute_difference_matrix"):
    # Compute the absolute difference matrix
    diff_matrix = np.abs(matrix2 - matrix1)

    plt.figure(figsize=(10, 10))

    # Plot the absolute difference matrix as a heatmap
    # Define the values range for the colorbar
    # othervalues for cmap are 'hot', 'jet', 'gray', 'viridis', 'magma', 'inferno', 'plasma'
    plt.imshow(diff_matrix, cmap='hot', vmin=0, vmax=1)

    # plt.imshow(diff_matrix, cmap='hot')
    plt.title(fileSaveName)
    plt.colorbar(label='Absolute Difference')

    # Hide axis ticks and labels
    plt.xticks([])
    plt.yticks([])

    # Save the plot as a PNG image
    plt.savefig(f'./results/{fileSaveName}.png', bbox_inches='tight')


# Defining the seed
np.random.seed(0)

# Defining the parameters
lambda_value = 0.1
num_iterations = 100

# Defining the dimensions
m = 10
n = 100
k = 50

# Defining the data set
# Dimensions: m x n
# X = np.array([
#                 [1, 3, 5, 2, 5, 7, 8, 9, 1, 2],
#                 [2, 4, 1, 3, 2, 1, 6, 12, 3, 4],
#                 [2, 4, 1, 3, 2, 0, -2, 8, 7, 9]
#             ]) 
X = np.random.randn(m, n)

# Dimensions: m x k
# D = np.random.randn(m, k)
# D /= np.linalg.norm(D, axis=0)

# create an identity matrix
D = np.eye(m, k)

# D = np.array([
#                 [1 / math.sqrt(5), 5 / math.sqrt(26)],
#                 [2 / math.sqrt(5), 1 / math.sqrt(26)]
#             ])

A = np.eye(k, k)
# Zero matrix of size k x k
B = np.zeros((m, k))
# B = np.random.randn(m, k)


def update_dictionary(a, b, dictionary, dict_cols):
    # For each column of the dictionary
    for j in range(dict_cols) :
        u_j = dictionary[:, j] + (b[:, j] - np.matmul(dictionary, a[:, j])) / a[j, j]
        dictionary[:, j] = u_j / max([1, np.linalg.norm(u_j)])

    return dictionary

# algorithm to draw i.i.d samples of p from a matrix
# Sample multiple columns from the matrix
def sample(matrix):
    num_cols = matrix.shape[1]
    while True :
        permutation = list(np.random.permutation(num_cols))
        for idx in permutation:
            yield matrix[:, idx].reshape(-1, 1)


def sample_columns(matrix, num_samples = 6):
    num_columns = matrix.shape[1]
    sample_indices = np.random.choice(num_columns, num_samples, replace=False)
    sampled_matrix = matrix[:, sample_indices]
    return sampled_matrix


# def sample(matrix):
#     num_cols = matrix.shape[1]
#     while True :
#         permutation = list(np.random.permutation(num_cols))
#         for idx in permutation:
#             yield matrix[:, idx].reshape(-1, 1)

objective_values = []
times = [i for i in range(1, num_iterations + 1)]

print()
print()

# X = sample(X)

for i in tqdm(range(num_iterations)):
    # print()
    # print("Iteration: ", i + 1)

    # Draw x t from p(x).
    # X_t = sample_random_column_from_matrix(X)
    samples = 10
    X_t = sample_columns(X, samples)

    # Define the optimization variables
    currAlpha = cp.Variable((k, samples))

    # Define the objective function
    objective = cp.Minimize((1 / 2) * cp.sum_squares(X_t - np.matmul(D, currAlpha)) + lambda_value * cp.norm(currAlpha, 1))

    constraints = []
    # Columns of D are normalized
    for j in range(k):
        constraints += [
            cp.norm(D[:, j]) <= 1,
        ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    optimalValue = problem.solve()

    # Retrieve the optimized alpha
    optimized_alpha = currAlpha.value


    A += (1/2) * np.matmul(optimized_alpha, optimized_alpha.T)
    B += np.matmul(X_t, optimized_alpha.T)

    D = update_dictionary(A, B, D, k)

    # get the optimal objective value
  
    
    objective_values.append(objective.value / (i + 1))

    x_hat = np.around(np.matmul(D, optimized_alpha), decimals=1)
    # plotDifferenceMatrix(X, x_hat, f"Matrix_Reconstruction_Difference_(Iteration{i + 1}, with K = {k})")
    # print(x_hat)
    # print(D)

print(objective_values[-1])

plt.close("all")
# Create new figure
plt.figure()
#  Plot the objective function with time
plt.plot(times[1:], objective_values[1:])
plt.xlabel('Time')
plt.ylabel('Objective function')
plt.show()
