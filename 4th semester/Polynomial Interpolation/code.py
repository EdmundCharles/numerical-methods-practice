import numpy as np
import matplotlib.pyplot as plt
from aux import table_function
from Lagrange import Lagrange




f = lambda x: x - 10*np.cos(x) + 3
g = lambda x: np.abs(x-10*np.cos(x) + 3)

def plot_1(f,a,b,xc,ns: list,alpha=1e-15):
    colors = ['red','orange','magenta','green','blue']
    if len(ns)<=5:
        for n in ns:
            col = colors[ns.index(n)]
            x_h, y_h = table_function(f,a,b,xc,n,alpha)
            ab = np.linspace(a,b,(b-a)*1000)
            p_val = np.array([Lagrange(x_h,y_h,x) for x in ab])
            plt.plot(ab,p_val,label = f'$L_{{{n-1}}}(x)$, {n} data points',color = col,linewidth = 1)
            plt.scatter(x_h,f(x_h),color = col)
        f_val = f(ab)
        plt.plot(ab,f_val,label = '$f(x)$',linewidth = 2)
        plt.legend(fontsize = 24)
        plt.tick_params(labelsize = 24)
        plt.grid()
        # plt.ylim(-1.1*max(f_val),1.5*max(f_val))
        plt.xlim(-10.1,-9.6)
        
        plt.show()



# plot_1(g,-10,10,0,[4,10,15,30],alpha=1e-15)


def plot_1_4(f,a,b,xc,n,alpha=1e-15):
        x_h, y_h = table_function(f,a,b,xc,n,alpha)
        ab = np.linspace(a,b,(b-a)*1000)
        p_val = np.array([Lagrange(x_h,y_h,x) for x in ab])
        plt.plot(ab,p_val,label = f'$L_{{{n-1}}}(x)$, {n} data points',linewidth = 1)
        plt.scatter(x_h,f(x_h))
        f_val = f(ab)
        plt.plot(ab,f_val,label = '$f(x)$',linewidth = 2)
        plt.legend(fontsize = 16)
        plt.tick_params(labelsize = 16)
        plt.grid()
        plt.ylim(-1.1*max(f_val),1.5*max(f_val))
        plt.show()



# plot_1_4(f,-10,10,0,[50],alpha=1e-16)




def plot_2(f,a,b,xc,ns: list,alpha=1e-15):
    colors = ['red','orange','magenta','green','blue']
    plt.rcParams.update({'font.size' : 16})
    if len(ns)<=5:    
        for n in ns: 
            col = colors[ns.index(n)] 
            x_h,y_h = table_function(f,a,b,xc,n,alpha)
            ab = np.linspace(a,b,(b-a)*1000)
            p_val = np.array([Lagrange(x_h,y_h,x) for x in ab])
            f_val = np.array(f(ab))
            deviations = abs(p_val - f_val)
            plt.plot(ab,deviations,label = f'$|f(x)-L_{{{n-1}}}(x)|$, {n} data points',color = col,linewidth = 0.5)
            plt.scatter(x_h,np.zeros(n),color = col)
        plt.ylabel('$|f(x) - p(x)|$')
        plt.xlabel('$x$')
        plt.legend()
        plt.semilogy()
        plt.grid()
        plt.show()
        
# plot_2(f,-10,10,0,[50],alpha=1e-15)


def plot_3(f,a,b,xc,n_start,n_end,alpha = 1e-15):
    ns = range(n_start,n_end+1)
    ab = np.linspace(a,b,(b-a)*100)
    f_val = np.array(f(ab))
    deviations = []
    plt.rcParams.update({'font.size' : 24})
    for n in ns:
        print(n)
        x_h,y_h = table_function(f,a,b,xc,n,alpha)
        p_val = np.array([Lagrange(x_h,y_h,x) for x in ab])
        deviations.append(np.max(np.abs(f_val - p_val)))
    plt.plot(ns,deviations,label = f'$x \\in [{a},{b}]$',marker = 'o',markersize = '3',linewidth = 1.2,color = 'black')
    plt.xlabel('$n$')
    plt.ylabel('max$|f(x)-L(x)|$')
    plt.grid()
    plt.semilogy()
    plt.legend()
    plt.show()

# plot_3(f,-10,10,0,3,70,alpha=1e-15)

def plot_stability_test(f, a, b, xc, n, alpha, epsilon=1e-3):
    """
    Тест на численную устойчивость.
    epsilon: амплитуда шума (вносимой погрешности)
    """
    # 1. Генерируем чистые данные
    x_nodes, y_clean = table_function(f, a, b, xc, n, alpha)
    
    # 2. Генерируем шум (случайная добавка от -epsilon до +epsilon)
    noise = np.random.uniform(-epsilon, epsilon, size=len(y_clean))
    y_noisy = y_clean + noise
    
    # 3. Строим полиномы на плотной сетке
    x_plot = np.linspace(a, b, 1000)
    
    # Полином на чистых данных
    p_clean = np.array([Lagrange(x_nodes, y_clean, x) for x in x_plot])
    # Полином на грязных данных
    p_noisy = np.array([Lagrange(x_nodes, y_noisy, x) for x in x_plot])
    
    # 4. Вычисляем "отклик" (разницу)
    response = np.abs(p_clean - p_noisy)
    
    # Оценка числа обусловленности (во сколько раз усилилась ошибка)
    amplification = np.max(response) / np.max(np.abs(noise))
    
    # --- Визуализация ---
    plt.figure(figsize=(10, 6))
    
    # Рисуем саму ошибку (отклик системы)
    plt.plot(x_plot, response, 'r-', label=f'Отклик на возмущение (Output Error)')
    
    # Рисуем уровень внесенного шума (горизонтальная линия)
    plt.axhline(y=epsilon, color='g', linestyle='--', label=f'Внесенный шум $\\varepsilon = {epsilon}$')
    
    plt.title(f'Тест на устойчивость (N={n}, $\\alpha={alpha}$)\nКоэффициент усиления ошибки $\\approx {amplification:.1f}$')
    plt.xlabel('x')
    plt.ylabel('Абсолютная разница |L_clean - L_noisy|')
    plt.yscale('log') # Логарифм, чтобы видеть масштабы катастрофы
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()

# ПРИМЕР ЗАПУСКА:
# Сравни два случая:
# 1. НЕУСТОЙЧИВЫЙ: Равномерная сетка (alpha мал), N=40. Шум усилится в 10^10 раз.
# plot_stability_test(f, -10, 10, 0, n=40, alpha=1e-5, epsilon=1e-4)

# 2. УСТОЙЧИВЫЙ: Сгущенная сетка (alpha=1.5), N=40. Шум усилится всего в 3-5 раз.
# plot_stability_test(f, -10, 10, 0, n=40, alpha=1.5, epsilon=1e-4)

import numpy as np
import matplotlib.pyplot as plt

# Не забудь импортировать свои функции
# from Lagrange import Lagrange
# from aux_tools import table_function

def plot_stability_side_by_side(f, a, b, xc, n, alpha, epsilon=1e-3):
    """
    Строит два графика рядом:
    1. Слева: Исходная функция + Полином (чистый) + Полином (зашумленный).
    2. Справа: Разница между чистым и зашумленным полиномом (Отклик).
    """
    # 1. Генерация данных
    x_nodes, y_clean = table_function(f, a, b, xc, n, alpha)
    
    # Добавляем шум (возмущение)
    np.random.seed(42) # Фиксируем seed для воспроизводимости
    noise = np.random.uniform(-epsilon, epsilon, size=len(y_clean))
    y_noisy = y_clean + noise
    
    # 2. Подготовка плотной сетки для отрисовки
    x_plot = np.linspace(a, b, 1000)
    f_val = f(x_plot)
    
    # Вычисляем полиномы
    p_clean = np.array([Lagrange(x_nodes, y_clean, x) for x in x_plot])
    p_noisy = np.array([Lagrange(x_nodes, y_noisy, x) for x in x_plot])
    
    # Разница (усиленная ошибка)
    difference = np.abs(p_clean - p_noisy)
    amplification = np.max(difference) / epsilon

    # 3. Построение графиков (1 строка, 2 колонки)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- ЛЕВЫЙ ГРАФИК: Сами функции ---
    ax1 = axs[0]
    ax1.plot(x_plot, f_val, 'k--', linewidth=1.5, alpha=0.6, label='f(x)')
    ax1.plot(x_plot, p_clean, 'b-', linewidth=1.5, label='$L_{clean}$')
    ax1.scatter(x_nodes, y_clean, color='blue', s=20, zorder=5, edgecolors='black', label='Nodes')
    ax1.plot(x_plot, p_noisy, 'r-', linewidth=1, alpha=0.8, label='$L_{perturbed}$')
    
    # Ограничим Y, если полином улетает в космос (для читаемости)
    limit = max(np.max(np.abs(f_val)), np.max(np.abs(p_clean))) 
    # Если "взрыв" слишком сильный, обрезаем, иначе график схлопнется
    if np.max(np.abs(p_noisy)) > 10 * limit:
        ax1.set_ylim(-2*limit, 2*limit)
    
    ax1.set_title(f'Interpolation ($N={n}, \\alpha={alpha}$)')
    ax1.set_xlabel('x')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- ПРАВЫЙ ГРАФИК: Ошибка (Разница) ---
    ax2 = axs[1]
    ax2.plot(x_plot, difference, 'r-', label='$|L_{clean} - L_{perturbed}|$')
    # Уровень шума для сравнения
    ax2.axhline(y=epsilon, color='b', linestyle='--', label='Noise Level')
    
    ax2.set_title(f'Stability Analysis')
    ax2.set_xlabel('x')
    ax2.set_yscale('log') # Логарифм обязателен, чтобы видеть порядки
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- ПРИМЕР ЗАПУСКА ---
# g = lambda x: np.abs(x - 4*np.cos(x) + 5)
plot_stability_side_by_side(f, -10, 10, -1.8, n=25, alpha=1e-15, epsilon=1e-4)

import numpy as np
import matplotlib.pyplot as plt

def plot_epsilon_dependency(f, a, b, xc, ns, alpha, epsilons):
    """
    Plots the dependency of the maximum output error on the input perturbation magnitude.
    
    Arguments:
    f        : function to interpolate
    a, b     : interval boundaries
    xc       : center of the grid (for clustering)
    ns       : list of polynomial degrees (number of nodes)
    alpha    : grid clustering parameter
    epsilons : list of perturbation magnitudes (e.g., [1e-10, 1e-5, 1e-1])
    """
    
    # Fine grid for evaluating the maximum deviation
    x_fine = np.linspace(a, b, 2000)
    
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size' : 16 })
    # Loop over different polynomial degrees
    for n in ns:
        max_deviations = []
        
        # 1. Generate the grid (fixed for a given n)
        x_nodes, _ = table_function(f, a, b, xc, n, alpha)
        
        # Loop over different epsilon values
        for eps in epsilons:
            # 2. Generate random noise vector delta with amplitude eps
            # We assume worst-case scenario or random uniform noise
            np.random.seed(42) # Fix seed for reproducibility across epsilons
            noise = np.random.uniform(-eps, eps, size=n)
            
            # 3. Calculate the response of the polynomial to this noise
            # Using linearity: L(y + noise) - L(y) = L(noise)
            # We construct a polynomial solely from the noise values
            response = np.array([Lagrange(x_nodes, noise, val) for val in x_fine])
            
            # 4. Find the maximum absolute deviation (Uniform norm)
            max_dev = np.max(np.abs(response))
            max_deviations.append(max_dev)
            

        plt.loglog(epsilons, max_deviations, '.-', label=f'N = {n}', linewidth=1.5)


    plt.loglog(epsilons, epsilons, 'k--', label='No perturbation', alpha=0.5)

    # Configuration
    plt.xlabel(r'Input Perturbation $\varepsilon$')
    plt.ylabel(r'Max Deviation')
    plt.title(f'Stability Analysis \n Grid parameter $\\alpha={alpha}$')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    plt.show()

ns_list = [7, 15,25, 40]
eps = np.logspace(-12,-3,10)
# plot_epsilon_dependency(g, -10, 10, 0, ns_list, alpha=1e-15, epsilons=eps)