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
    plt.title('Absolute Difference Matrix')
    plt.colorbar(label='Absolute Difference')

    # Hide axis ticks and labels
    plt.xticks([])
    plt.yticks([])

    # Display the plot
    # plt.show()

    # Save the plot as a PNG image
    plt.savefig(f'{fileSaveName}.png', bbox_inches='tight')


# Defining the seed
np.random.seed(0)

# Defining the parameters
lambda_value = 0.1
num_iterations = 5

# Defining the dimensions
m = 3
n = 10
k = 2

# Defining the data set
# Dimensions: m x n
X = np.array([
                [1, 3, 5, 2, 5, 7, 8, 9, 1, 2],
                [2, 4, 1, 3, 2, 1, 6, 12, 3, 4],
                [2, 4, 1, 3, 2, 0, -2, 8, 7, 9]
            ]) 


print(X)

# Dimensions: m x k
# D = np.random.randn(m, k)
# D /= np.linalg.norm(D, axis=0)

# create an identity matrix
D = np.eye(m, k)

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

print()
print()

for i in tqdm(range(num_iterations)):
    print()
    print("Iteration: ", i + 1)

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

    D = update_dictionary(A, B, D, k)

    x_hat = np.around(np.matmul(D, optimized_alpha), decimals=1)
    plotDifferenceMatrix(X, x_hat, f"absolute_difference_matrix_{i + 1}")
    print(x_hat)
    # print(D)


plt.close("all")
# Create new figure
plt.figure()
#  Plot the objective function with time
plt.plot(times[1:], objective_values[1:])
plt.xlabel('Time')
plt.ylabel('Objective function')
# Only show the new figure
plt.show()
