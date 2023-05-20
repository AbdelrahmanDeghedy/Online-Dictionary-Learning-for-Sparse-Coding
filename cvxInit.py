import cvxpy as cp
import numpy as np

# Generate some random data
np.random.seed(1)
m = 3
k = 4
n = 10
X = np.random.rand(m, n)

# Initialize the dictionary 
D = X[:, 0:k]
magnitude = np.linalg.norm(D,2)
D = D / magnitude
# Initialize the matrices A and B to zero
A = np.zeros((k, k))
B = np.zeros((m, k))
alpha2 = np.zeros((k, n))
iterations = 100
jj = np.zeros(iterations)

# Loop over the data points
for i in range(iterations):
    # Select a data point
    jo = np.random.randint(n)
    x = X[:, jo]
    
    # Solve the L1-norm minimization problem
    alpha = cp.Variable(k)
    obj = cp.sum_squares(cp.norm(x - D @ alpha,"fro" ) )+ 0.1 * cp.norm(alpha, 1)
    constraints = [cp.norm(D, 2) <= 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    print("Optimization status:", prob.status)
    alpha2[:, jo] = alpha.value
    
    # Update the matrices A and B
    A += alpha.value.reshape(-1, 1) @ alpha.value.reshape(1, -1)
    B += x.reshape(-1, 1) @ alpha.value.reshape(1,-1)
    
    # Update D
    for j in range(k):
        # Compute the new j-th atom
        u = ((B[:, j] - D @ A[:, j]) / A[j, j]) + D[:, j]
        D[:, j] = u / max(np.linalg.norm(u, 2), 1)


    magnitude = np.linalg.norm(D,2)
    D = D / magnitude   
    jj[i] = np.linalg.norm(X - D @ alpha2, 'fro')
    
import matplotlib.pyplot as plt

# Plot the convergence curve
t = np.arange(iterations)
plt.plot(t, jj)
plt.xlabel('Iteration')
plt.ylabel('L2-norm error')
plt.title('Convergence curve of K-SVD algorithm')
plt.show()