import numpy as np
import matplotlib.pyplot as plt

def matrixWalk(P):
    state = P[0]
    states = [0]
    states_dict = {}
    num_states = np.shape(state)[0]
    for c in range(n):
        next_state = np.random.choice(num_states, p=state)
        state = P[next_state]
        states.append(next_state)

    for s in states:
        states_dict.setdefault(s, 0)
        states_dict[s] = states_dict[s] + 1

    return states_dict


def plot():
    # Histogram (PDF)
    states = np.array(sorted(dist.keys()))
    print("STATES: ", states)
    probs = np.array([dist[s] / n for s in states])

    plt.figure()
    plt.bar(states, probs)
    plt.title("Empirical PMF of U (Simulation by matrix walk)")
    plt.xlabel("State U")
    plt.ylabel("Probability")
    plt.grid(True)

    # CCDF
    ccdf = 1 - np.cumsum(probs)
    ccdf[ccdf <= 0] = 1e-7  # avoid log(0)

    plt.figure()
    plt.plot(states, ccdf, marker='o', linestyle='--')
    plt.yscale('log')
    plt.title("CCDF of U (log scale) (Simulation by matrix walk)")
    plt.xlabel("State U")
    plt.ylabel("P(U > u)")
    plt.grid(True, which='both', linestyle='--')

    plt.show()



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


# GENERATE MATRIX USING THE FOWARD ALGORITHM TASK 6.2.2
# All states
for i in range(U_max):
    # All combinations of a and b
    for a_index, a in enumerate(A_values):
        for b_index, b in enumerate(B_values):
            next_state = max(min(i + b, U_max) - a, 0)  # Add to currentt state i to calculate next state
            prob = A_probs[a_index] * B_probs[b_index]  # Probability of next state
            P[i][next_state] += prob

print(P)
# TASK 6.2.3 Markov simulation
dist = matrixWalk(P)
print(dist)
plot()



