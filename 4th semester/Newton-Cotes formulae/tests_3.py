import numpy as np
import matplotlib.pyplot as plt
from integrator import Integrator
from scipy.integrate import quad
from mpmath import mp

#Ezka
f_ez = lambda x: x**2
#Elliptic integral
alpha = 179.9
k = np.sin(np.radians(alpha) / 2) 
f_elliptic = lambda phi: 1 / np.sqrt(1 - 0.99999 * np.sin(phi)**2)
f_ell = lambda phi: 1 / mp.sqrt(1 - k**2 * mp.sin(phi)**2)
a_E,b_E = 0.1,10
#Runge's function
f_runge  = lambda x: 1/(1+25*x**2)
# f_runge = lambda x: 

a_R, b_R = -1,1
#Integrators
elliptic_integrator = Integrator(f_elliptic,a_E,b_E)
# ell_integrator = Integrator(f_ell,a_E,b_E)
runge_integrator = Integrator(f_runge,a_R,b_R)

easy_integrator = Integrator(f_ez,0,10)
#Test 1: accuracy test.
def test_1(integrator):
    epses = np.logspace(-13,-2,13)
    # e_val,_ = integrator.exact_val()
    e_val = mp.quad(f_ell,[0.1,10])
    errors = []
    for eps in epses:
        s, i = integrator.trapz_naive(eps)
        err = abs(e_val - s)
        errors.append(err)
    #Fontsize
    plt.rcParams.update({'font.size': 16})
    #Plots
    plt.plot(epses,errors,'o', color = 'blue',label = 'Abs error')
    plt.plot(epses,epses, linestyle = '--', color = 'red',label = 'Bound')
    #Settings
    plt.grid(True)
    plt.legend()
    plt.loglog()
    plt.xlabel("Desired accuracy")
    plt.ylabel("Absoulte error")
    #Show
    plt.show()

#Test 2: computational difficulty.
def test_2(integrator):
    epses = np.logspace(-15,-2,13)
    calls_for_naive = []
    calls_for_efficent = []
    for eps in epses:
        _, calls_naive = integrator.trapz_naive(eps)
        _, calls_efficient = integrator.trapz_efficent(eps)
        calls_for_naive.append(calls_naive)
        calls_for_efficent.append(calls_efficient)
    #Acceleration
    A = calls_for_naive[-1]/calls_for_efficent[-1]
    #Fontsize
    plt.rcParams.update({'font.size': 16})
    #Plots
    plt.plot(epses,calls_for_naive, marker = 'o', color = 'red',label = 'Naive trapezoidal')
    plt.plot(epses,calls_for_efficent, marker = 'o', color = 'blue',label = 'Efficent trapezoidal')
    #Settings
    plt.legend()
    plt.semilogx()
    # plt.semilogy(base = 2)
    plt.xlabel("Desired accuracy")
    plt.ylabel("Times $f(x)$ was called")
    plt.grid(True)
    plt.title(f'Acceleration A = {A}')
    #Show
    plt.show()

# test_1(elliptic_integrator)
# test_2(runge_integrator)


def test_1(integrator):
    a, b = integrator.a, integrator.b
    
    mp.dps = 60 
    

    e_val_mp = mp.ellipf(b, k**2) - mp.ellipf(a, k**2)
    e_val = float(e_val_mp) 
    
    epses = np.logspace(-12, -2, 13)
    errors = []
    
    print(f"Analytical e_val: {e_val}")
    
    for eps in epses:
        s, i = integrator.trapz_naive(eps)
        err = abs(e_val - s)
        errors.append(err)
        print(f"eps: {eps:.1e} | i: {i} | err: {err:.2e}")

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
    plt.plot(epses, errors, 'o', color='blue', label='Abs error')
    plt.plot(epses, epses, linestyle='--', color='red', label='Bound (Ideal)')
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.loglog()
    plt.xlabel("Desired accuracy (eps)")
    plt.ylabel("Absolute error")
    plt.title(f"Accuracy Test (k={k:.6f})")
    plt.show()
# test_1(elliptic_integrator)

def testik():
    errors = []
    e_val = round(43.466483741691666047133114330549386038552878527319975274351,16)
    epses = np.logspace(-15, -2, 13)
    for eps in epses:
        s, _ = quad(f_elliptic,0,10,epsabs=eps)
        print(s)
        err = abs(e_val - s)
        print(err)
        errors.append(err)
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
    plt.plot(epses, errors, 'o', color='blue', label='Abs error')
    plt.plot(epses, epses, linestyle='--', color='red', label='Bound (Ideal)')
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.loglog()
    plt.xlabel("Desired accuracy (eps)")
    plt.ylabel("Absolute error")
    plt.title(f"Accuracy Test (k={k:.6f})")
    plt.show()



# def test_stability_smooth(integrator):    
    # mp.dps = 60
    # exact, _ = integrator.exact_val()
    
    # epses = np.logspace(-15, -5, 25)
    # err_trapz = []
    # err_neight = []

    # print("Running Smooth Function Stability Test...")

    # for eps in epses:
    #     s_n,_ = integrator.trapz_efficent(eps)
    #     err_trapz.append(abs(exact - s_n))
    #     s_n, _ = integrator.N_C_n(eps)
    #     err_neight.append(abs(exact - s_n))

    # plt.figure(figsize=(10, 6))
    # plt.loglog(epses, err_trapz, '-o', label='Trapezoidal', color='blue')
    # plt.loglog(epses, err_neight, '-s', label='Newton-Cotes n=9', color='red')
    # plt.loglog(epses,epses,'--',label = 'Bound')
    
    
    # plt.xlabel("Desired Precision (eps)")
    # plt.ylabel("Absolute Error")
    # plt.title(r"Stability Test")
    # plt.grid(True, which="both", alpha=0.3)
    # plt.ylim(1e-16)
    # plt.legend()
    # plt.show()

def test_stability(integrator):
    exact =  integrator.mp_exact_val()
    n_start, n_end = 2, 25
    ns = range(n_start,n_end+1)
    errors = []
    for n in ns:
        result = integrator.S_n(n)
        print(result)
        error = abs(exact - result)
        errors.append(error)
    # plt.semilogy(ns,errors,'-o',label = 'Absolute error')
    plt.plot(ns,errors,'-o',label = 'Absolute error for Runge function')
    plt.xlabel('Number of nodes $n$',fontsize = 16)
    plt.ylabel('Absolute error',fontsize = 16)
    plt.title(f'N-C formulae for $n$ from ${{{n_start}}}$ to ${{{n_end}}}$')
    plt.grid()
    plt.xticks(ns)
    plt.legend(fontsize = 16)
    

    plt.show()

test_stability(runge_integrator)