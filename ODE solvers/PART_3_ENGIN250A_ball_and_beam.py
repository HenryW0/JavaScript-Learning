from scipy.integrate import odeint
from control import lqr
import matplotlib.pyplot as plt
import numpy as np

g = 9.81 #gravitational constant in meters per second squared
r = 0.0254  #radius of gear in meters
L = 0.4255 #Length of beam in meters
T = 0.025 #Delay/time constant in seconds
K = 1.5 #rad/sV

A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]) #System matrix for s variable change
B = np.array([0, 0, 0, 1]).reshape(-1, 1) #Matrix for control inputs
#print(B, np.shape(B))

Q = np.diag([2, 10, 1, 5]) #State Weight/Importance Matrix
R = np.array([[0.01]]) #Control Cost Matrix, only 1 by 1 since there is a single control input

K_m = -1 * lqr(A, B, Q, R)[0][0]
#print(K_m)

def ball_beam(x, t):
    #print(x)
    z, v, theta, omega = x
    a_x = (5*g)/7 * (r/L) * ((-omega)/T * np.cos(theta) - (omega) ** 2 * np.sin(theta))
    b_x = (5*g)/7 * (r/L) * (K/T) * np.cos(theta)
    
    #s1 is equal to z, s2 is equal to v
    s3 = (5*g)/7 * (r/L) * np.sin(theta)
    s4 = (5*g)/7 * (r/L) * (omega) * np.cos(theta)
    s = np.array([z, v, s3, s4]).reshape(-1, 1)
    #print(s, np.shape(s))

    vu = np.matmul(K_m, s)[0]
    u = (1/b_x) * (-a_x + vu) #vu is -k1s1 - k2s2 - k3s3 - k4s4

    #print("Control", u)
    dxdt = [v, (5*g)/7 * (r/L) * np.sin(theta) - (5/7) * ((L/2) - z) * (r/L)**2 * (omega)**2 * (np.cos(theta)**2), omega, (-omega/T) + ((K*u)/T)]
    return dxdt

x0 = [0.15, -0.42, 50 * np.pi/180, 0]
#x0 = [0.15, -0.30, 50 * np.pi/180, -0.5]
t = np.linspace(0, 15, 1001)

sol = odeint(ball_beam, x0, t, args=())

if np.any(abs(sol[:, 0]) >= 0.19):
    print("Collision imminent at edge")

if np.any(abs(sol[:, 2]) >= np.deg2rad(60)):
    print("Servo angle out of operation range")

plt.title(f'Initial Conditions: {[round(i, 3) for i in x0]}')
plt.plot(t, sol[:, 0], 'b-', label='z')
plt.plot(t, sol[:, 1], 'g--', label='z dot')
plt.plot(t, sol[:, 2], 'r-.', label = 'theta')
plt.plot(t, sol[:, 3], ':', color = 'magenta', label = 'theta dot')

plt.legend(loc='best')
plt.ylabel('Units dependent on state variable')
plt.xlabel('t (seconds)')
plt.grid()
plt.show()

plt.plot(sol[:,0], (180/np.pi) * sol[:,2])
plt.title("z vs theta")
plt.xlabel('Z (position) in Meters')
plt.ylabel('Theta (angle) in Degrees')
plt.grid()
plt.show()