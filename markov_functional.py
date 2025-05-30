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

dist = {}
dist.setdefault(U_next, 0)


def calculateUNext(U_prev):
    a_sample = np.random.choice(A_values, p=A_probs)
    b_sample = np.random.choice(B_values, p=B_probs)
    U = max(min(U_prev + b_sample, U_max) - a_sample, 0)
    return U

def plot():
    # Histogram (PDF)
    states = np.array(sorted(dist.keys()))
    probs = np.array([dist[s] / n for s in states])

    plt.figure()
    plt.bar(states, probs)
    plt.title("Empirical PMF of U")
    plt.xlabel("State U")
    plt.ylabel("Probability")
    plt.grid(True)

    # CCDF
    ccdf = 1 - np.cumsum(probs)

    plt.figure()
    plt.plot(states, ccdf, marker='o', linestyle='--')
    plt.yscale('log')
    plt.title("CCDF of U (log scale)")
    plt.xlabel("State U")
    plt.ylabel("P(U > u)")
    plt.grid(True, which='both', linestyle='--')

    plt.show()


for i in range(n):
    U_next = calculateUNext(U_next)
    dist.setdefault(U_next, 0)
    dist[U_next] = dist[U_next] + 1



print(dist)
plot()



