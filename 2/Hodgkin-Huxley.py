"""
This script simulates a basic Hodgkin-Huxley model based on their original values
"""
import matplotlib.pyplot as plt
import numpy as np

K = 0
Na = 1
R = 2

# ------ Simulation constants

dt = 0.01

# ------ Channel parameters

G = [0] * 3
G[K] = 36
G[Na] = 120
G[R] = 0.3

E = [0] * 3
E[K] = -12
E[Na] = 115
E[R] = 10.613

# ------ Environment constants
I_ext = 0
V = -10

t_rec = 0
x = np.zeros((3,), dtype=np.float32)
Alpha = np.zeros((3,), dtype=np.float32)
Beta = np.zeros((3,), dtype=np.float32)
I = np.zeros((3,), dtype=np.float32)
conductances = np.zeros((3,), dtype=np.float32)
x[2] = 1

V_plot = []
# ------ Integration time

for t in np.arange(-30, 50, dt):
    if np.isclose(t, 10):
        I_ext = 10
    if np.isclose(t, 40):
        I_ext = 0

    Alpha[K] = (10 - V) / (100 * (np.exp((10 - V) / 10) - 1))
    Alpha[Na] = (25 - V) / (10 * (np.exp((25 - V) / 10) - 1))
    Alpha[R] = 0.07 * np.exp(-V / 20)

    Beta[K] = 0.125 * np.exp(-V / 80)
    Beta[Na] = 4 * np.exp(-V / 18)
    Beta[R] = 1 / (np.exp((30 - V) / 10) + 1)

    tau = 1 / (Alpha + Beta)

    x_0 = Alpha * tau

    x = (1 - dt / tau) * x + dt / tau * x_0

    # Calculate conductances
    conductances[K] = G[K] * x[K] ** 4
    conductances[Na] = G[Na] * x[Na] ** 3 * x[R]
    conductances[R] = G[R]

    # Ohm's Law
    I[K] = conductances[K] * (V - E[K])
    I[Na] = conductances[Na] * (V - E[Na])
    I[R] = conductances[R] * (V - E[R])

    # Update voltage
    V += dt * (I_ext - np.sum(I))
    if t > 0.0:
        V_plot.append(V)

plt.plot(np.arange(0, 50, dt), V_plot)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage")
plt.show()
