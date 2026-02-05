import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 3*x - 5*np.cos(x) - 1
n = 15
net = np.linspace(0,100,n)
ab = np.linspace(0,100,n*100)
y = f(net)
coeffs = np.polyfit(net,y,deg=len(y)-1)
p = np.polyval(coeffs,ab)
print(p[1])
def plot_1(): 
    plt.plot(ab,p,label = '$p(x)$')
    plt.plot(ab,f(ab),label = '$f(x)$')
    plt.scatter(net,f(net),label = 'Data')
    plt.legend()
    plt.grid()
    plt.show()

errors = []

for i in range(len(ab)):
    errors.append(abs(f(ab[i]) - p[i]))

def plot_2():
    f_val = np.array(f(ab))
    errors = abs(p-f_val)
    plt.plot(ab,errors)
    plt.ylabel('$|f(x) - p(x)|$')
    plt.xlabel('$x$')
    plt.scatter(net,np.zeros(n))
    plt.show()

ns = range(1,100)
def plot_3():
    errors_max = []
    for j in ns:
        net = np.linspace(0,100,j)
        y = f(net)
        coeffs = np.polyfit(net,y,deg=len(y)-1)
        p = np.polyval(coeffs,ab)
        f_val = np.array(f(ab))
        diff = abs(p - f_val)
        errors_max.append(max(diff))
    print(diff)
    plt.plot(ns,errors_max)
    plt.xlabel('$n$')
    plt.loglog()
    plt.ylabel('max$|f(x)-p(x)|$')
    plt.show()

plot_1()
