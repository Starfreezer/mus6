import numpy as np
import matplotlib.pyplot as plt

U_max = 30
n = 10 ** 6
U_0 = 0
U_next = 0

np.random.seed(1337)

# Inter-arrival Time values and probs
A_values = [2, 3, 4]
A_probs = [0.2, 0.6, 0.2]

# Batch sizes values and probs
B_values = [1, 2, 3]
B_probs = [0.6, 0.3, 0.1]

P = np.zeros(shape=(U_max, U_max))


# All states
for i in range(U_max):
    # All combinations of a and b
    for a_index, a in enumerate(A_values):
        for b_index, b in enumerate(B_values):
            next_state = max(min(i + b, U_max) - a, 0) # Add to currentt state i to calculate next state
            prob = A_probs[a_index] * B_probs[b_index] # Probability of next state
            P[i][next_state] += prob


print(P)