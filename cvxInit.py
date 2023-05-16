
import math
import cvxpy as cp
import numpy as np

# Define the initial values
# Dt = np.array([1 / math.sqrt(5), 2 / math.sqrt(5)])
# Xt = np.array([1, 3, 5, 2])
# print("Dt:", Dt.shape)
# print("Xt:", Xt.shape)



lambda_value = 0.1
num_iterations = 10
m = 2; n = 4; k = 2

X = np.array([
                [1, 3, 5, 2],
                [2, 4, 1, 3]
            ]) 

D = np.array([
                [1 / math.sqrt(5), 5 / math.sqrt(26)],
                [2 / math.sqrt(5), 1 / math.sqrt(26)]
            ])

A = np.random.randn(k, k)
B = np.random.randn(m, k)

# Define the variables
alpha = cp.Variable(k)


for i in range(num_iterations):
    for j in range(k):
        # Access the j-th column of X, D
        Xt = X[:, j]
        Dt = D[:, j]

        # Define the objective function
        objective = cp.Minimize(cp.sum_squares(Xt - Dt @ alpha) + lambda_value * cp.norm(alpha, 1))

        # Define the problem
        problem = cp.Problem(objective)

        # Solve the problem
        problem.solve()

        # Retrieve the optimized alpha
        optimized_alpha = alpha.value

        print(A[:, j].shape)

        A[:, j] += (1/2) * np.outer(optimized_alpha, optimized_alpha.T)
        B[:, j] += (1/2) * np.outer(Xt, optimized_alpha.T)

        print("A:\n", A)
        # print("Optimized alpha:", optimized_alpha)
