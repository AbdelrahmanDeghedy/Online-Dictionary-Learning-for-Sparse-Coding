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

# Dimensions: m x n
X = np.random.randn(m, n)


def update_dictionary(a, b, dictionary, dict_cols):
    # For each column of the dictionary
    for j in range(dict_cols) :
        u_j = dictionary[:, j] + (b[:, j] - np.matmul(dictionary, a[:, j])) / a[j, j]
        dictionary[:, j] = u_j / max([1, np.linalg.norm(u_j)])

    return dictionary

# algorithm to draw i.i.d samples of p from a matrix
def sample_columns(matrix, num_samples = 6):
    num_columns = matrix.shape[1]
    sample_indices = np.random.choice(num_columns, num_samples, replace=False)
    sampled_matrix = matrix[:, sample_indices]
    return sampled_matrix, sample_indices


def func(samples, num_iterations):
    # Defining the data set
    # Dimensions: m x n
    X = np.random.randn(m, n)

    # Dimensions: m x k
    # D = np.random.randn(m, k)
    # D /= np.linalg.norm(D, axis=0)

    # create an identity matrix
    D = X[:, 0 : k]
    D /= np.linalg.norm(D, axis=0)

    A = np.eye(k, k)
    # Zero matrix of size k x k
    B = np.zeros((m, k))
    # B = np.random.randn(m, k)
    objective_values = []
    totalAlpha = np.zeros((k, n))


    for i in tqdm(range(num_iterations)):
        # Draw x t from p(x).
        X_t, selectedIndices = sample_columns(X, samples)

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
        totalAlpha[:, selectedIndices] = optimized_alpha


        A += (1/2) * np.matmul(optimized_alpha, optimized_alpha.T)
        B += np.matmul(X_t, optimized_alpha.T)

        D = update_dictionary(A, B, D, k)    
        
        objective_values.append(objective.value / (i + 1))
        # objective_values.append( (1 / (i + 1)) * np.linalg.norm(X - np.matmul(D, totalAlpha), 2) ** 2 + lambda_value * np.linalg(currAlpha, 1) )


        x_hat = np.around(np.matmul(D, optimized_alpha), decimals=1)
        # plotDifferenceMatrix(X, x_hat, f"Matrix_Reconstruction_Difference_(Iteration{i + 1}, with K = {k})")
    return objective_values

plt.close("all")
# Create new figure
plt.figure()
#  Plot the objective function with time


# labels = ['ahmed', 'hamada', 'deghedy']
patchSizes = [1, 4, 10]
labels = ["Our Method", f"Batch n = {patchSizes[1]}", f"Batch n = {patchSizes[2]}"]
times = [i for i in range(1, num_iterations + 1)]


for i, (label, samples) in enumerate(zip(labels, patchSizes)):
    objective_values = func(samples, num_iterations)
    plt.plot(times, objective_values, label=label)

plt.xlabel('Time')
plt.ylabel('Objective function')
plt.legend()
plt.show()
