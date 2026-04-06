import numpy as np
import matplotlib.pyplot as plt
from RadauMethod import Integrator
from aux import Integrator_NC


def test_1_side_by_side(integrator1, integrator2, eps_min1=-13, eps_min2=-13, title1="Function 1", title2="Function 2"):
    print("Started test 1 (Side-by-Side)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.rcParams.update({'font.size': 18})
    
    configs = [(integrator1, eps_min1, ax1, title1), 
               (integrator2, eps_min2, ax2, title2)]
    
    for integrator, eps_min, ax, title in configs:
        NC = Integrator_NC(integrator.f, integrator.a, integrator.b)
        epses = np.logspace(eps_min, -2, -eps_min -1)
        
        e_val, _ = NC.exact_val()
        
        errors = []
        errors_NC = []
        
        for eps in epses:
            s, _ = integrator.radau_3(eps)
            err = abs(e_val - s)
            errors.append(err)
            
            s, _ = NC.trapz_efficent(eps)
            err = abs(e_val - s)
            errors_NC.append(err)

        # Plots
        ax.plot(epses, errors, '-o', color='blue', label='Abs error, Radau', linewidth=0.5)
        ax.plot(epses, errors_NC, '-o', color='red', label='Abs error, NC Trapz', linewidth=0.5)
        ax.plot(epses, epses, linestyle='--', color='green', label='Bound')
        
        # Settings
        ax.grid(True)
        ax.legend(fontsize = 22)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(epses)
        
        # # Опциональный поворот подписей, чтобы не накладывались
        ax.tick_params(axis='both', labelsize = 18) 
        
        ax.set_xlabel("Desired accuracy", fontsize = 22)
        ax.set_ylabel("Absolute error", fontsize = 22)
        ax.set_title(title)
        
    plt.tight_layout()
    plt.show()

def test_2_side_by_side(integrator1, integrator2, eps_min1=-13, eps_min2=-13, title1="Function 1", title2="Function 2"):
    print("Started test 2 (Side-by-Side)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.rcParams.update({'font.size': 16})
    
    configs = [(integrator1, eps_min1, ax1, title1), 
               (integrator2, eps_min2, ax2, title2)]
    
    for integrator, eps_min, ax, title in configs:
        NC = Integrator_NC(integrator.f, integrator.a, integrator.b)
        epses = np.logspace(eps_min, -2, -eps_min -1)
        
        calls = []
        calls_NC = []
        
        for eps in epses:
            _, calls_temp = integrator.radau_3(eps)
            calls.append(calls_temp)
            
            _, calls_temp = NC.trapz_efficent(eps)
            calls_NC.append(calls_temp)

        # Plots
        ax.plot(epses, calls, marker='o', color='blue', label='Radau, $n = 3$', linewidth=0.5)
        ax.plot(epses, calls_NC, marker='o', color='red', label='NC Trapz', linewidth=0.5)
        
        # Settings
        ax.grid(True)
        ax.legend(fontsize = 22)
        ax.set_xscale('log')
        ax.set_yscale('log', base=10)
        ax.set_xticks(epses)
        
        ax.tick_params(axis='both', labelsize = 18)
        
        ax.set_xlabel("Desired accuracy",fontsize = 22)
        ax.set_ylabel("Times $f(x)$ was called",fontsize = 22)
        ax.set_title(title)
        
    plt.tight_layout()
    plt.show()

def test_4_side_by_side(integrator1, integrator2, log_N_max1=3, log_N_max2=3, title1="Function 1", title2="Function 2"):
    print("Started test 4 (Side-by-Side)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.rcParams.update({'font.size': 14})
    
    configs = [(integrator1, log_N_max1, ax1, title1), 
               (integrator2, log_N_max2, ax2, title2)]
    
    for integrator, log_N_max, ax, title in configs:
        NC = Integrator_NC(integrator.f, integrator.a, integrator.b)
        
        N_intervals = np.unique(np.logspace(0, log_N_max, 20, dtype=int))
        e_val, _ = NC.exact_val() 
        
        h_values = []
        errors_radau = []
        errors_trapz = []
        
        for N in N_intervals:
            h = (integrator.b - integrator.a) / N
            h_values.append(h)
            
            # Радо
            intervals = np.linspace(integrator.a, integrator.b, N + 1)
            s_rad = np.sum([integrator._s_radau_3(intervals[j], intervals[j+1]) for j in range(N)])
            errors_radau.append(max(abs(e_val - s_rad), 1e-16)) 
            
            # Трапеции
            s_nc, h_nc, _ = NC._S2(N + 1)
            errors_trapz.append(max(abs(e_val - s_nc), 1e-16))

        # Plots
        ax.plot(h_values, errors_radau, '-o', color='blue', label='Radau, $n = 3$', linewidth=1.5)
        ax.plot(h_values, errors_trapz, '-o', color='red', label='NC Trapz', linewidth=1.5)
        
        # Асимптоты (взято из вашего исходника)
        C_rad = errors_radau[2] / (h_values[2]**5)
        C_trap = errors_trapz[2] / (h_values[2]**2)
        
        ax.plot(h_values, [C_rad * h**5 for h in h_values], '--', color='cyan', label='Bound $\mathcal{O}(h^5)$')
        ax.plot(h_values, [C_trap * h**(2) for h in h_values], '--', color='orange', label='Bound $\mathcal{O}(h^2)$')
        
        # Settings
        ax.grid(True)
        ax.legend(fontsize = 22)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.invert_xaxis() 

        ax.set_xlabel("Grid step $h$ (decreasing $\\rightarrow$)", fontsize = 22)
        ax.set_ylabel("Absolute error", fontsize = 22)
        ax.set_title(title)
        
    plt.tight_layout()
    plt.show()

func = lambda x: 1/(1+25*x**2)
a, b = 0,1
runge_integrator = Integrator(func,a,b)

ns = lambda x: abs(1 - 2/(1+25*x**2))

a, b = 0, 1
ns_integrator = Integrator(ns,a,b)


# test_1_side_by_side(runge_integrator, ns_integrator, eps_min1=-15, eps_min2=-12, title1="Runge Function", title2="NS Function")
test_2_side_by_side(runge_integrator, ns_integrator, eps_min1=-15, eps_min2=-12, title1="Runge Function", title2="NS Function")
test_4_side_by_side(runge_integrator, ns_integrator, log_N_max1=3, log_N_max2=4,title1="Runge Function", title2="NS Function")