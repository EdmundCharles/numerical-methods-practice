import numpy as np
import matplotlib.pyplot as plt
from RadauMethod import Integrator

def test_1(integrator):
    epses = np.logspace(-16,-2,13)
    e_val = integrator.exact_val()
    errors = []
    for eps in epses:
        s,_ = integrator.radau_3(eps)
        err = abs(e_val - s)
        errors.append(err)
    #Fontsize
    plt.rcParams.update({'font.size': 16})
    #Plots
    plt.plot(epses,errors,'--o', color = 'blue',label = 'Abs error')
    plt.plot(epses,epses, linestyle = '--', color = 'red',label = 'Bound')
    #Settings
    plt.grid(True)
    plt.legend()
    plt.loglog()
    plt.xlabel("Desired accuracy")
    plt.ylabel("Absoulte error")
    #Show
    plt.show()

def test_2(integrator):
    epses = np.logspace(-15,-2,13)
    calls = []
    for eps in epses:
        _, calls_temp = integrator.radau_3(eps)
        calls.append(calls_temp)
    #Fontsize
    plt.rcParams.update({'font.size': 16})
    #Plots
    plt.plot(epses,calls, marker = 'o', color = 'red',label = 'Radau, $n = 3$')
    #Settings
    plt.legend()
    plt.semilogx()
    # plt.semilogy(base = 2)
    plt.xlabel("Desired accuracy")
    plt.ylabel("Times $f(x)$ was called")
    plt.grid(True)
    #Show
    plt.show()


func = lambda x: 1/(1+25*x**2)
a, b = -1,1

runge_integrator = Integrator(func,a,b)

test_1(runge_integrator)
test_2(runge_integrator)