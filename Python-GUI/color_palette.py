import numpy as np
import time as time
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use("TkAgg")

#A + Bcos(2pi(Cx+D))
'''
A is brightness
B is constrast
C controls how quickly colors change (frequency)
D controls the exact location of the color picks
x goes from 0 to 1
'''

matrices = {'A': np.array([0.2, 0.3, 0.48]), 'B': np.array([0.35, 0.45, 0.18]),
            'C': np.array([1.0, 1.0, 1.0]), 'D': np.array([0.6, 0.43, 0.42])}

fig = plt.figure()


def plot_gradient():
    start = time.time()

    brightness = matrices['A']
    contrast = matrices['B']
    freq = matrices['C']
    phase_shift = matrices['D']
    
    s = R = G = B = np.linspace(0, 1, 1000)
    RGB = np.array([R, G, B])
    
    for n in range(3):
        RGB[n] = np.clip(brightness[n] + contrast[n] * np.cos(2 * np.pi * (freq[n] * RGB[n] + phase_shift[n])), 0, 1)
    
    for m, n in enumerate(s):
        plt.axvline(n, color = (RGB[0][m], RGB[1][m], RGB[2][m]))
        plt.draw()
    
    print(time.time() - start)
    return s, RGB

def slider_update(name, value):
    matrices[name[0]][int(name[1])] = value
    

root = tk.Tk()
root.config(bg="black")
root.title("Color Palette")
root.protocol('WM_DELETE_WINDOW', root.quit)

sliders = tk.Frame(root)
sliders.pack()

A0 = tk.Scale(sliders, label= 'Brightness Red', fg='Red', length = 200, from_=0.0, to=1.0, resolution = 0.1, tickinterval=0.2,
                 orient="horizontal", variable = tk.DoubleVar(), command = lambda x: slider_update('A0', x))
A0.pack()

B0 = tk.Scale(sliders, label= 'Contrast Red', fg='Red', length = 200, from_=0.0, to=1.0, resolution = 0.1, tickinterval=0.2,
                 orient="horizontal", variable = tk.DoubleVar(), command = lambda x: slider_update('B0', x))
B0.pack()


btn = tk.Button(root, text = "Update", command = plot_gradient)
btn.pack()


canvas = FigureCanvasTkAgg(fig, master=root) # Convert the Figure to a tkinter widget
canvas.get_tk_widget().pack() # Show the widget on the screen

root.geometry("800x600+10+10")
root.mainloop() # Start tkinter's mainloop
