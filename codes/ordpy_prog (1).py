# ---------------------------------------------------------
# Toy Example: Complexity–Entropy (CH) Plane Demonstration
# Time series: Periodic, Chaotic, Stochastic
# Uses ordpy.complexity_entropy
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import ordpy

# ---------------------------------------------------------
# 1. Generate time series
# ---------------------------------------------------------

N = 5000
t = np.linspace(0, 50, N)

# (a) Periodic signal (simple oscillation)
periodic = np.sin(2 * np.pi * 0.2 * t)

# (b) Chaotic signal (Logistic map)
r = 3.9
chaotic = np.zeros(N)
chaotic[0] = 0.2
for i in range(1, N):
    chaotic[i] = r * chaotic[i-1] * (1 - chaotic[i-1])

# (c) Stochastic signal (Gaussian white noise)
stochastic = np.random.normal(0, 1, N)

# ---------------------------------------------------------
# 2. Compute Permutation Entropy and Statistical Complexity
# ---------------------------------------------------------

dx = 5   # embedding dimension
taux = 1  # delay

H_periodic, C_periodic = ordpy.complexity_entropy(periodic, dx=dx, taux=taux)
H_chaotic, C_chaotic = ordpy.complexity_entropy(chaotic, dx=dx, taux=taux)
H_stochastic, C_stochastic = ordpy.complexity_entropy(
    stochastic, dx=dx, taux=taux)

# ---------------------------------------------------------
# 3. Plot CH plane
# ---------------------------------------------------------

plt.figure(figsize=(7, 6), dpi=100)

plt.scatter(H_periodic, C_periodic, s=120, label='Periodic (Sine)', marker='o')
plt.scatter(H_chaotic, C_chaotic, s=120,
            label='Chaotic (Logistic map)', marker='s')
plt.scatter(H_stochastic, C_stochastic, s=120,
            label='Stochastic (Noise)', marker='^')

# annotate points
plt.text(H_periodic+0.01, C_periodic, "Periodic")
plt.text(H_chaotic+0.01, C_chaotic, "Chaotic")
plt.text(H_stochastic+0.01, C_stochastic, "Stochastic")

plt.xlabel("Normalized Permutation Entropy (H)")
plt.ylabel("Statistical Complexity (C)")
plt.title("Complexity–Entropy (CH) Plane")

plt.legend(loc='right')
plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 4. Print values
# ---------------------------------------------------------

print("Permutation Entropy and Statistical Complexity")
print("----------------------------------------------")
print(f"Periodic   : H = {H_periodic:.4f}, C = {C_periodic:.4f}")
print(f"Chaotic    : H = {H_chaotic:.4f}, C = {C_chaotic:.4f}")
print(f"Stochastic : H = {H_stochastic:.4f}, C = {C_stochastic:.4f}")
