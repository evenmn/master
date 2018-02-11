import matplotlib.pyplot as plt
import numpy as np

time_list = [1e-3,1e-4,1e-5]
error_A = [7.20780980969e-05, 7.06735820562e-05, 7.05334975555e-05]
error_B = [0.000134434302129,0.000142788024292,0.000143492061876]
error_C = [0.0233015137256,0.0233140343757,0.0233152830048]
error_D = [0,0,0]

#---Plot---
SZ = {'size':'16'}

plt.plot(np.log10(time_list),np.log10(error_A),marker='o')
plt.plot(np.log10(time_list),np.log10(error_B),marker='o')
plt.plot(np.log10(time_list),np.log10(error_C),marker='o')
plt.plot(np.log10(time_list),np.log10(error_D),marker='o')
plt.legend(['Method A','Method B','Method C','Method D'],loc='best')
plt.xlabel('Time step, $\log_{10}(dt)$',**SZ)
plt.ylabel('Absolute error, $\log_{10}({\cal O})$',**SZ)
plt.axis([-2.5,-5.5,-1,-5])
plt.grid()
plt.savefig('error_plot.png')
plt.show()
