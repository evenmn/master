import numpy as np
import matplotlib.pyplot as plt


T  = 10000
dt = 0.1
dx = 0.1
N  = int(T/dt)

move = np.random.random(N)
move_disc = np.where(move>0.5,1,-1)

x = np.zeros(N)
sumu = 0
for i in range(N):
    sumu += move_disc[i]
    x[i] = sumu
    
t = np.linspace(0,10,N)
plt.plot(x, t)
plt.show()

