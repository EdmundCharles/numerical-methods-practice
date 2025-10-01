import matplotlib.pyplot as plt
import numpy as np

#Algebraic
f1 = lambda x: x**4-3*x**3+3*x**2-2
df1 = lambda x: 4*x**3-9*x**2+6*x
phi1 = lambda x: (3*x**3-3*x**2+2)**0.25
dphi1 = lambda x: (x*(-1.5+2.25*x))/(2-3*x**2+3*x**3)**0.75

#Transcendential
f2 = lambda x: 3*(np.log10(x))**2+6*np.log(1+x) -3
phi2 = lambda x: np.e**(0.5*(1-(np.log10(x))**2))-1
dphi2 = lambda x: np.e**(0.5*(1-(np.log10(x))**2))*0.5*(-2*np.log10(x))*(1/(x*np.log(10)))

#Test
ff = lambda x: (3-x)**(1/3)
dff = lambda x: -1/3*(3-x)**(-2/3)
#Plotting

x = np.linspace(0,2,1000)
xx = np.linspace(0,2.5,1000)
# plt.plot(x,df1(x),linestyle = 'dashed',label = r"$f_1'(x)$")
plt.plot(x,f1(x),label = '$f_1(x)$')
# plt.plot([-2.189,-0.449],[0,0],'-o',color = 'green')
# plt.plot([0.449,4],[0,0],'-o',color = 'green',label = 'Границы корней')
# plt.plot(x,f2(x),label = '$f_2(x)$')
# plt.plot(x,abs(phi1(x)),label = '$|\\varphi _1(x)|$')
# plt.plot(x,abs(dphi1(x)),label = r"$|\varphi_1'(x)|$",linestyle = 'dashed')
# plt.plot(x,abs(phi2(x)),label = '$|\\varphi _2(x)|$')
# plt.plot(x,abs(dphi2(x)),label = r"$|\varphi_2'(x)|$",linestyle = 'dashed')
# plt.axvline(0,linewidth = '0.5',color = 'black')
# plt.axhline(0,linewidth = '0.5',color = 'black')
# plt.plot(xx,abs(ff(xx)),label = '$|\\varphi(x)|$')
# plt.plot(xx,abs(dff(xx)),label = r"$|\varphi'(x)|$",linestyle = 'dashed')
plt.grid(which='major',linestyle = '-')
plt.grid(which='minor',linestyle = '-')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.minorticks_on()
# plt.xlim(-3,5)
# plt.ylim(-3,5)
plt.legend()
plt.show()