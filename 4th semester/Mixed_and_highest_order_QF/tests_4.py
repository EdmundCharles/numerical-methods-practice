import numpy as np
import matplotlib.pyplot as plt
from RadauMethod import Integrator
from aux import Integrator_NC


def test_1(integrator):
    print(f"Started test 1")
    NC = Integrator_NC(integrator.f,integrator.a,integrator.b)
    epses = np.logspace(-13,-2,12)
    e_val = integrator.exact_val()
    errors = []
    errors_NC = []
    for eps in epses:
        s,_ = integrator.radau_3(eps)
        err = abs(e_val - s)
        errors.append(err)
        s,_ = NC.trapz_efficent(eps)
        print(s)
        err = abs(e_val - s)
        errors_NC.append(err)
    #Fontsize
    plt.rcParams.update({'font.size': 16})
    #Plots
    plt.plot(epses,errors,'-o', color = 'blue',label = 'Abs error, Radau',linewidth = 0.5)
    plt.plot(epses,errors_NC,'-o', color = 'red',label = 'Abs error, NC Trapz',linewidth = 0.5)
    plt.plot(epses,epses, linestyle = '--', color = 'green',label = 'Bound')
    #Settings
    plt.grid(True)
    plt.legend()
    plt.loglog()
    plt.xticks(epses)
    plt.xlabel("Desired accuracy")
    plt.ylabel("Absoulte error")
    #Show
    plt.show()

def test_2(integrator):
    print(f"Started test 2")
    NC = Integrator_NC(integrator.f,integrator.a,integrator.b)
    epses = np.logspace(-13,-2,12)
    calls = []
    calls_NC = []
    for eps in epses:
        _, calls_temp = integrator.radau_3(eps)
        calls.append(calls_temp)
        _, calls_temp = NC.trapz_efficent(eps)
        calls_NC.append(calls_temp)
    #Fontsize
    plt.rcParams.update({'font.size': 16})
    #Plots
    plt.plot(epses,calls, marker = 'o', color = 'blue',label = 'Radau, $n = 3$',linewidth = 0.5)
    plt.plot(epses,calls_NC,marker = 'o', color = 'red',label = 'NC Trapz',linewidth = 0.5)
    #Settings
    plt.legend()
    plt.semilogx()
    plt.semilogy(base = 10)
    plt.xticks(epses)
    plt.xlabel("Desired accuracy")
    plt.ylabel("Times $f(x)$ was called")
    plt.grid(True)
    #Show
    plt.show()

# def test_3(integrator):
#     print(f"Started test 3")
#     NC = Integrator_NC(integrator.f, integrator.a, integrator.b)

#     epses = np.logspace(-13, -2, 12) 
#     e_val = integrator.exact_val()
    

#     calls_radau = []
#     errors_radau = []
    
#     calls_NC = []
#     errors_NC = []
    
#     for eps in epses:
#         s_rad, c_rad = integrator.radau_3(eps)
#         err_rad = abs(e_val - s_rad)
#         calls_radau.append(c_rad)
#         errors_radau.append(err_rad)
        
#         s_nc, c_nc = NC.trapz_efficent(eps)
#         err_nc = abs(e_val - s_nc)
#         calls_NC.append(c_nc)
#         errors_NC.append(err_nc)

#     # Fontsize
#     plt.rcParams.update({'font.size': 16})
    
#     # Plots
#     plt.plot(calls_radau, errors_radau, '-o', color='blue', label='Radau, $n = 3$', linewidth=0.5)
#     plt.plot(calls_NC, errors_NC, '-o', color='red', label='NC Trapz', linewidth=0.5)
    
#     # Settings
#     plt.grid(True)
#     plt.legend()
#     plt.loglog() 
#     plt.xlabel("Times $f(x)$ was called")
#     plt.ylabel("Absolute error")
    
    # Show
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def test_4(integrator):
    print(f"Started test 4")
    N_values = np.unique(np.logspace(0, 3, 20, dtype=int))
    e_val = integrator.exact_val()
    
    h_values = []
    errors_radau = []
    errors_trapz = []
    
    for N in N_values:
        h = (integrator.b - integrator.a) / N
        h_values.append(h)
        
        intervals = np.linspace(integrator.a, integrator.b, N + 1)
        s_rad = np.sum([integrator._s_radau_3(intervals[j], intervals[j+1]) for j in range(N)])
        errors_radau.append(max(abs(e_val - s_rad), 1e-16))
        

        x_trapz = intervals
        y_trapz = integrator.f(x_trapz)
        s_nc = np.trapz(y_trapz, x_trapz)
        errors_trapz.append(max(abs(e_val - s_nc), 1e-16))

    # Fontsize
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 7))
    
    
    plt.plot(h_values, errors_radau, '-o', color='blue', label='Radau, $n = 3$', linewidth=1.5)
    plt.plot(h_values, errors_trapz, '-o', color='red', label='NC Trapz', linewidth=1.5)
    


    C_rad = errors_radau[2] / (h_values[2]**5)
    C_trap = errors_trapz[2] / (h_values[2]**2)
    
    plt.plot(h_values, [C_rad * h**5 for h in h_values], '--', color='cyan', label='Теория $\mathcal{O}(h^5)$')
    plt.plot(h_values, [C_trap * h**2 for h in h_values], '--', color='orange', label='Теория $\mathcal{O}(h^2)$')
    
    # Settings
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.loglog() 
    
    plt.gca().invert_xaxis() 
    
    plt.xlabel("Grid step $h$ (decreasing $\\rightarrow$)")
    plt.ylabel("Absolute error")
    plt.title("Convergence Order: Radau vs Trapezoidal")
    
    # Show
    plt.show()

def test_4(integrator):
    NC = Integrator_NC(integrator.f, integrator.a, integrator.b)
    
    N_intervals = np.unique(np.logspace(0, 3, 20, dtype=int))
    
    e_val, _ = NC.exact_val() 
    
    h_values = []
    errors_radau = []
    errors_trapz = []
    
    for N in N_intervals:
        h = (integrator.b - integrator.a) / N
        h_values.append(h)
        #######
        intervals = np.linspace(integrator.a, integrator.b, N + 1)
        s_rad = np.sum([integrator._s_radau_3(intervals[j], intervals[j+1]) for j in range(N)])
        errors_radau.append(max(abs(e_val - s_rad), 1e-16)) 
        #######
        s_nc, h_nc, _ = NC._S2(N + 1)
        errors_trapz.append(max(abs(e_val - s_nc), 1e-16))

    # Fontsize
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 7))
    
    plt.plot(h_values, errors_radau, '-o', color='blue', label='Radau, $n = 3$', linewidth=1.5)
    plt.plot(h_values, errors_trapz, '-o', color='red', label='NC Trapz', linewidth=1.5)
    
    C_rad = errors_radau[2] / (h_values[2]**5)
    C_trap = errors_trapz[2] / (h_values[2]**2)
    
    plt.plot(h_values, [C_rad * h**5 for h in h_values], '--', color='cyan', label='Bound $\mathcal{O}(h^5)$')
    plt.plot(h_values, [C_trap * h**(1+2/3) for h in h_values], '--', color='orange', label='Bound $\mathcal{O}(h^2)$')
    
    # Settings
    plt.grid()
    plt.legend()
    plt.loglog() 
    
    plt.gca().invert_xaxis() 

    plt.xlabel("Grid step $h$")
    plt.ylabel("Absolute error")
    plt.title("Convergence Order: Radau vs Trapezoidal")
    
    # Show
    plt.show()
func = lambda x: 1/(1+25*x**2)
a, b = -1,1
runge_integrator = Integrator(func,a,b)

ns = lambda x: abs(1 - 2/(1+25*x**2))

a, b = 0, 1
ns_integrator = Integrator(ns,a,b)


# test_1(runge_integrator)
# test_2(runge_integrator)
# test_3(runge_integrator)
# test_4(runge_integrator)



test_1(ns_integrator)
# test_2(ns_integrator)
# test_3(ns_integrator)
# test_4(ns_integrator)

# Nc = Integrator_NC(ns,0,2*np.pi)
# print(Nc.trapz_efficent(1e-20))
# print(Nc.exact_val())