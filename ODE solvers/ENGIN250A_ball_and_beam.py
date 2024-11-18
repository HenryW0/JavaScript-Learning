from scipy.integrate import odeint
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import numpy as np

g = 9.81 #gravitational constant in meters per second squared
r = 0.0254  #radius of gear in meters
L = 0.4255 #Length of beam in meters
T = 0.025 #Delay/time constant in seconds
K = 1.5 #rad/sV

A = np.array([[0, 1, 0, 0], [0, 0, (5 * g * r)/(7*L), 0], [0, 0, 0, 1], [0, 0, 0, -1/T]]) #Linearized System Matrix
B = np.array([0, 0, 0, K/T]).reshape(-1, 1) #Matrix for control inputs

#print(B, np.shape(B))

#K_m = np.array([1, 0, 6.6, 10]) #K matrix, different than K motor constant
#u is no longer equal to 0 since we have a control input

poles = np.array([-2, -4, -6, -8])
placement = place_poles(A, B, poles)
K_m = placement.gain_matrix
#print(K_m)

def ball_beam(x, t):
    #print(x)
    z, v, theta, omega = x
    u = np.matmul(-K_m, x)[0]

    #print("Control", u)
    dxdt = [v, (5*g)/7 * (r/L) * np.sin(theta) - (5/7) * ((L/2) - z) * (r/L)**2 * (omega)**2 * (np.cos(theta)**2), omega, (-omega/T) + ((K*u)/T)]
    return dxdt

x0 = [0.05, -0.22, 40 * np.pi/180, 0.4]
t = np.linspace(0, 10, 1001)

sol = odeint(ball_beam, x0, t, args=())

if np.any(abs(sol[:, 0]) >= 0.19):
    print("Collision imminent at edge")

if np.any(abs(sol[:, 2]) >= np.deg2rad(60)):
    print("Servo angle out of operation range")

plt.title(f'Initial Conditions: {[round(i, 3) for i in x0]}')
plt.plot(t, sol[:, 0], 'b', label='z')
plt.plot(t, sol[:, 1], 'g', label='z dot')
plt.plot(t, sol[:, 2], 'r', label = 'theta')
plt.plot(t, sol[:, 3], color = 'magenta', label = 'theta dot')

plt.legend(loc='best')
plt.ylabel('Units dependent on state variable')
plt.xlabel('t (seconds)')
plt.grid()
plt.show()

plt.plot(sol[:,0], (180/np.pi) * sol[:,2])
plt.title("z vs theta")
plt.grid()
plt.show()