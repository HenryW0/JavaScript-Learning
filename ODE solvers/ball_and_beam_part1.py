from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

g = -9.81 #gravitational constant in meters per second squared
r = 0.2  #radius of gear in meters
L = 1 #Length of beam in meters
T = 0.25 #Delay constant in seconds
#U is equal to 0 due to no control input so this is ignored

def ball_beam(x, t):
    z, v, theta, omega = x
    dxdt = [v, (5*g)/7 * (r/L) * np.sin(theta) - (5/7) * ((L/2) - z) * (r/L)**2 * (omega)**2 * (np.cos(theta)**2), omega, -omega/T]
    return dxdt

x0 = [0, 0.1, 0, 0.1]
t = np.linspace(0, 10, 1001)

sol = odeint(ball_beam, x0, t, args=())

plt.title(f'Initial Conditions: {x0}')
plt.plot(t, sol[:, 0], 'b', label='z')
plt.plot(t, sol[:, 1], 'g', label='z dot')
plt.plot(t, sol[:, 2], 'r', label = 'theta')
plt.plot(t, sol[:, 3], color = 'magenta', label = 'theta dot')

plt.legend(loc='best')
plt.ylabel('Units dependent on state variable')
plt.xlabel('t (seconds)')
plt.grid()
plt.show()

plt.plot(sol[:,0], sol[:,1])
plt.title("z vs z dot")
plt.grid()
plt.show()