import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from integrator import Integrator
import mpmath as mp
# Ns = [2**i+1 for i in range(0,5)]
# y = 0
# for n in Ns:
#     x = np.linspace(0,1,n)
#     print(len(x))
#     plt.scatter(x,np.ones(n)*y)
#     plt.text(-0.13,y-0.05,f'N = {n - 1}',fontsize = 20)
#     y+=1
# # plt.xlim(-1,1)
# plt.axis('off')
# plt.show()

# alpha = 179.9
# k = np.sin(np.radians(alpha) / 2) 
# f_elliptic = lambda phi: 1 / np.sqrt(1 - k**2 * np.sin(phi)**2)
# a_E,b_E = 0.1 , 10
# x = np.linspace(a_E,b_E,1000)
# y = f_elliptic(x)
# plt.rcParams.update({'font.size': 16})
# plt.plot(x,y,label = 'Elliptic function',color = 'red')
# plt.grid()
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.title(f'k = {round(k,10)}')
# plt.show()

# x = np.array([np.cos((np.pi*(2*i+1))/6) for i in range(0,3)])
# f = lambda x: 1/(x+2)
# y = f(x)
# print(x)
# print(print(y))
# print(2/(-np.sqrt(3) + 4))

f = lambda x: 1/(1+25*x**2)
inte = Integrator(f,-1,1)

print(mp.quad(f,[0,1]))