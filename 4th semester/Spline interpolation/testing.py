import numpy as np
import matplotlib.pyplot as plt
from Cubic_spline import cubic_spline
from aux_func import Lagrange, table_function


import numpy as np
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt

def plot_1(f, d2f, a, b, xc, ns: list, alpha=1e-15):
    ab = np.linspace(a, b, int((b - a) * 1000))
    f_val = f(ab)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    colors = ['red', 'orange', 'magenta', 'green']
    
    for i, n in enumerate(ns):
        ax = axs[i]
        col = colors[i]
        

        x_h, y_h = table_function(f, a, b, xc, n, alpha)
        
        spline_vals, ab_spline = cubic_spline(x_h, y_h, sd_a=d2f(a), sd_b=d2f(b))
        

        ax.plot(ab, f_val, label='Function', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        

        if i == 3:    
            p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
            ax.plot(ab, p_val, label=f'$L_{{{n-1}}}(x)$', color=col, linewidth=2)
            

        if i != 3:    
            ax.plot(ab_spline, spline_vals, label='Cubic Spline', color='blue', linewidth=2)
        

        ax.scatter(x_h, f(x_h), color=col, s=20, zorder=5)
        

        ax.set_title(f'Number of nodes $N = {n}$', fontsize=14)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=12)
        ax.grid(True)
        
        limit = max(np.abs(f_val))
        ax.set_ylim(-1.2 * limit, 1.5 * limit)

    plt.tight_layout()
    plt.show()

def plot_2(f, d2f, a, b, xc, ns: list, alpha=1e-15):
    plt.rcParams.update({'font.size': 14})
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    colors = ['red', 'orange', 'magenta', 'green']
    ab = np.linspace(a, b, int((b - a) * 200))
    f_val = f(ab)
    
    for i, n in enumerate(ns):
        ax = axs[i]
        col = colors[i]
        
        x_h, y_h = table_function(f, a, b, xc, n, alpha)
        

        p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
        dev_lagrange = np.abs(p_val - f_val)
        

        spline_vals, ab_spline = cubic_spline(x_h, y_h, sd_a=d2f(a), sd_b=d2f(b))
        dev_spline = np.abs(spline_vals - f(ab_spline))
        

        if i == 3:    
            ax.plot(ab, dev_lagrange, label=f'Lagrange Error ($N={n}$)', color=col, linewidth=1.5)
        
        if i != 3:    
            ax.plot(ab_spline, dev_spline, label='Spline Error', color='blue', linewidth=1.5)
        

        ax.scatter(x_h, np.zeros(n) + 1e-16, color=col, s=20, marker='x', label='Nodes')
        

        ax.set_title(f'Absolute error for $N={n}$', fontsize=14)
        ax.set_xlabel('$x$')
        ax.set_ylabel('Absolute Error')
        ax.semilogy()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_3(f, d2f, a, b, xc, n_start, n_end, alpha=1e-15):
    ns = range(n_start, n_end + 1,10)
    ab = np.linspace(a, b, int((b - a) * 100))
    
    f_val = np.array(f(ab))
    
    dev_lagrange = []
    dev_spline = []
    
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
    
    for n in ns:
        x_h, y_h = table_function(f, a, b, xc, n, alpha)
        

        # p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
        # dev_lagrange.append(np.max(np.abs(f_val - p_val)))
        

        spline_vals, ab_spline = cubic_spline(x_h, y_h, sd_a=d2f(a), sd_b=d2f(b))
        dev_spline.append(np.max(np.abs(f(ab_spline) - spline_vals)))
        

    # plt.plot(ns, dev_lagrange, label='Lagrange max error', marker='o', markersize=4, linewidth=1.5, color='black')
    plt.plot(ns, dev_spline, label='Spline max error', linewidth=1.5, color='blue')
    
    plt.title(f'Max Error $x \\in [{a}, {b}]$', fontsize=16)
    plt.xlabel('$N$ (number of nodes)')
    plt.ylabel('$\\max |f(x) - P(x)|$')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.semilogy()
    plt.legend()
    plt.show()





import numpy as np
import matplotlib.pyplot as plt

def plot_spline_stability(f, d2f, a, b, xc, n, alpha, epsilon=1e-4):


    x_nodes, y_clean = table_function(f, a, b, xc, n, alpha)
    
    sda_clean = d2f(a)
    sdb_clean = d2f(b)
    
    np.random.seed(42)
    noise_y = np.random.uniform(-epsilon, epsilon, size=len(y_clean))
    noise_sda = np.random.uniform(-epsilon, epsilon)
    noise_sdb = np.random.uniform(-epsilon, epsilon)
    
    y_noisy = y_clean + noise_y
    sda_noisy = sda_clean + noise_sda
    sdb_noisy = sdb_clean + noise_sdb
    
    spline_clean_vals, spline_clean_x = cubic_spline(x_nodes, y_clean, sda_clean, sdb_clean)
    spline_noisy_vals, spline_noisy_x = cubic_spline(x_nodes, y_noisy, sda_noisy, sdb_noisy)
    

    diff_spline = np.abs(spline_clean_vals - spline_noisy_vals)


    x_plot = np.linspace(a, b, 1000)
    f_val = f(x_plot)


    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    

    ax1 = axs[0]
    ax1.plot(x_plot, f_val, 'k--', linewidth=1.5, alpha=0.6, label='f(x)')
    ax1.plot(spline_clean_x, spline_clean_vals, 'b-', linewidth=1.5, label='$S_{clean}$')
    ax1.plot(spline_noisy_x, spline_noisy_vals, 'r-', linewidth=1.5, alpha=0.8, label='$S_{perturbed}$')
    ax1.scatter(x_nodes, y_clean, color='blue', s=20, zorder=5, edgecolors='black', label='Nodes')
    
    ax1.set_title(f"Cubic Spline Interpolation ($N={n}, \\alpha={alpha}$)")
    ax1.set_xlabel('x')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)


    ax2 = axs[1]
    ax2.plot(spline_clean_x, diff_spline, 'r-', linewidth=1.5, label='$|S_{clean} - S_{perturbed}|$')
    

    ax2.axhline(y=epsilon, color='b', linestyle='--', label=f'Perturbation level $\\epsilon = {epsilon}$')
    
    ax2.set_title("Stability Analysis")
    ax2.set_xlabel('x')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()

# plot_spline_stability(f,d2f,a,b,xc,70,alpha=1e-15)


def plot_spline_epsilon_dependency(f, a, b, xc, ns, alpha, epsilons):
    """
    Plots the dependency of the maximum output error on the input perturbation magnitude
    for Cubic Spline Interpolation.
    """
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size' : 16 })
    
    for n in ns:
        max_deviations = []
        

        x_nodes, _ = table_function(f, a, b, xc, n, alpha)
        num_nodes = len(x_nodes)
        
        for eps in epsilons:

            np.random.seed(42) 
            noise_y = np.random.uniform(-eps, eps, size=num_nodes)
            noise_sda = np.random.uniform(-eps, eps)
            noise_sdb = np.random.uniform(-eps, eps)
            

            spline_vals, _ = cubic_spline(x_nodes, noise_y, sd_a=noise_sda, sd_b=noise_sdb)
            
            max_dev = np.max(np.abs(spline_vals))
            max_deviations.append(max_dev)
            

        plt.loglog(epsilons, max_deviations, '.-', label=f'N = {n}', linewidth=1.5)


    plt.loglog(epsilons, epsilons, 'k--', label='Perturbation level $\\varepsilon$', alpha=0.5)

    # Оформление графика
    plt.xlabel(r'Input Perturbation $\varepsilon$')
    plt.ylabel(r'Max Deviation')
    plt.title(f'Spline Stability Analysis \n Grid parameter $\\alpha={alpha}$')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    plt.show()
eps = np.logspace(-12,-2,10)

ns2 = [10,30,70,1000]
# plot_spline_epsilon_dependency(f,a,b,xc,ns2,alpha=1e-15,epsilons=eps)



##########


def plot_3_add(f, d2f,d4f, a, b, xc, n_start, n_end, alpha=1e-15):
    ns = range(n_start, n_end + 1)
    ab = np.linspace(a, b, int((b - a) * 1000))
    il = b-a
    hs  = np.array([il/(n+1) for n in ns])
    d4fmax = np.max(np.abs(np.array([d4f(x) for x in ab])))
    f_val = np.array(f(ab))
    
    dev_lagrange = []
    dev_spline = []
    
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
    
    for n in ns:
        x_h, y_h = table_function(f, a, b, xc, n, alpha)

        # p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
        # dev_lagrange.append(np.max(np.abs(f_val - p_val)))
        

        spline_vals, ab_spline = cubic_spline(x_h, y_h, sd_a=d2f(a), sd_b=d2f(b))
        dev_spline.append(np.max(np.abs(f(ab_spline) - spline_vals)))
        

    # plt.plot(ns, dev_lagrange, label='Lagrange max error', marker='o', markersize=4, linewidth=1.5, color='black')
    plt.plot(hs, dev_spline, label='Spline max error', linewidth=1.5, color='blue')
    plt.plot(hs,d4fmax*5/384*hs**4,label = '$Ch^4, \; C = max|f^{(4)}(x)|$')
    
    plt.title(f'Max Error $x \\in [{a}, {b}]$', fontsize=16)
    plt.xlabel('$h$, grid spacing')
    plt.ylabel('$\\max |f(x) - P(x)|$')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    # plt.semilogy()
    plt.loglog()
    plt.legend()
    plt.show()
#-------------------#
f = lambda x: x - 4*np.cos(x) + 5 
d2f = lambda x: 4*np.cos(x)
d4f = lambda x: -4*np.cos(x)
r = lambda x: 1/(1+25*x**2)
d2r = lambda x: -(1+25*x**2)**(-2)*50*x
d4r = lambda x: 15000 * (1 - 250*x**2 + 3125*x**4) / (1 + 25*x**2)**5
g = lambda x: np.abs(f(x))
d2g = lambda x: -d2f(x) if x<0 else d2f(x)
d4g = lambda x: -d4f(x) if x<0 else d4f(x)
#--------------#
a , b = -10,10
xc = 0
ns = [10,20,31,15]
#------------------#
# plot_1(g,d2g,a,b,xc,ns)

# plot_2(g,d2g,a,b,xc,ns)

# plot_3(g,d2g,a,b,0,5,1000)


# plot_3_add(r,d2r,d4r,a,b,xc,5,1000,alpha=1e-16)
# plot_3_add(f,d2f,d4f,-10,10,xc,5,1000,alpha=1e-16)
plot_3_add(g,d2g,d4g,a,b,xc,5,1000)