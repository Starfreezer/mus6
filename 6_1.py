import numpy as np

P = np.array([
    [0.1, 0.2, 0.3, 0.4],  
    [0.0, 0.0, 1.0, 0.0],  
    [0.0, 0.0, 0.0, 1.0],  
    [1.0, 0.0, 0.0, 0.0]
])

P_modified = np.array([
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0]
])

# Initial x_0 = [1, 0, 0, 0]
x = np.array([1.0, 0.0, 0.0, 0.0])

print("Transition matrix P")
print(f"x_0 = {x}")

# 8.1.4 Use the recurrence x_{i+1} = x_i @ P to compute x_1, x_2, ..., x_{10}
for i in range(1, 11):
    x = x @ P
    print(f"x_{i} = {x}")

# 8.1.5 If series of x_i converges, the resulting limit is equal to the average state distribution.
# left-hand eigenvector
eigenvalues, eigenvectors = np.linalg.eig(P.T)
# x_s (P - I) = 0
idx = np.argmin(np.abs(eigenvalues - 1.0))
v = eigenvectors[:, idx].real
stationary = v / np.sum(v)

print("Stationary state distribution:")
print(stationary)

print("\n###########################################################\n")
print("Modified transition matrix P_modified")
# Initial x_0 = [1, 0, 0, 0]
x = np.array([1.0, 0.0, 0.0, 0.0])
print(f"x_0 = {x}")
for i in range(1, 11):
    x = x @ P_modified
    print(f"x_{i} = {x}")

# If series of x_i converges, the resulting limit is equal to the average state distribution.
# left-hand eigenvector
eigenvalues, eigenvectors = np.linalg.eig(P_modified.T)
# x_s (P - I) = 0
idx = np.argmin(np.abs(eigenvalues - 1.0))
v = eigenvectors[:, idx].real
stationary = v / np.sum(v)

print("Stationary state distribution:")
print(stationary)