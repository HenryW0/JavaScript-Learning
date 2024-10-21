from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

delta = 0.05
gamma = 0.4
omega = 1.3


def duffing(x, t):
    p, v = x
    dxdt = [v, gamma * np.cos(omega*t) - (delta * v) + p - (p**3)]
    return dxdt

x0 = [7, -0.8]
t = np.linspace(0, 50, 1001)

sol = odeint(duffing, x0, t, args=())

plt.plot(t, sol[:, 0], 'b', label='y')
plt.plot(t, sol[:, 1], 'g', label='y dot')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.plot(sol[:,0], sol[:,1])
plt.title("y vs y dot")
plt.grid()
plt.show()