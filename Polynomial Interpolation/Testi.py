import numpy as np
import matplotlib.pyplot as plt
from aux import table_function
from Lagrange import Lagrange

f = lambda x: x - 4*np.cos(x) + 5 
# f = lambda x: np.abs(x - np.cos(x) + 3 )
g = lambda x: np.abs(f(x))


def plot_1(f, a, b, xc, ns: list, alpha=1e-15):
    # Заранее считаем точную функцию для отрисовки фона на всех графиках
    ab = np.linspace(a, b, int((b - a) * 1000))
    f_val = f(ab)
    
    # Создаем сетку графиков 2x2 (так как у нас 4 значения n)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Превращаем матрицу 2x2 в плоский список для удобного цикла
    
    colors = ['red', 'orange', 'magenta', 'green']
    
    # Итерируемся одновременно по n, цветам и осям (графикам)
    for i, n in enumerate(ns):
        ax = axs[i]
        col = colors[i]
        
        # --- Твои вычисления ---
        x_h, y_h = table_function(f, a, b, xc, n, alpha)
        p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
        
        # --- Рисование на конкретной оси ax ---
        # 1. Рисуем исходную функцию g(x) (пунктиром для сравнения)
        ax.plot(ab, f_val, label='$g(x)$', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 2. Рисуем полином Лагранжа
        ax.plot(ab, p_val, label=f'$L_{{{n-1}}}(x)$', color=col, linewidth=2)
        
        # 3. Рисуем узлы
        ax.scatter(x_h, f(x_h), color=col, s=40, zorder=5)
        
        # --- Оформление конкретного графика ---
        ax.set_title(f'Number of nodes $N = {n}$', fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(True)
        # Ограничение по Y (чуть изменил, чтобы брался максимум по модулю для симметрии)
        limit = max(np.abs(f_val))
        ax.set_ylim(-1.2 * limit, 1.5 * limit)

    plt.tight_layout()  # Чтобы подписи не наезжали друг на друга
    plt.show()

# Пример вызова для 4 значений:



def plot_2(f, a, b, xc, ns: list, alpha=1e-15):
    # Настройка шрифтов
    plt.rcParams.update({'font.size': 14})
    
    # Создаем сетку графиков 2x2
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()  # Плоский список осей для удобного цикла
    
    colors = ['red', 'orange', 'magenta', 'green']
    
    # Плотная сетка для отрисовки кривых
    ab = np.linspace(a, b, int((b - a) * 200))
    f_val = f(ab)
    
    for i, n in enumerate(ns):
        ax = axs[i]
        col = colors[i]
        
        # 1. Вычисляем узлы и значения
        x_h, y_h = table_function(f, a, b, xc, n, alpha)
        
        # 2. Вычисляем значения полинома (используем ту же логику, что в plot_1)
        p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
        
        # 3. Считаем модуль ошибки
        deviations = np.abs(p_val - f_val)
        
        # --- Рисование на конкретной оси ax ---
        # График ошибки
        ax.plot(ab, deviations, label=f'Error ($N={n}$)', color=col, linewidth=1.5)
        
        # Точки узлов (где ошибка = 0)
        ax.scatter(x_h, np.zeros(n) + 1e-16, color=col, s=20, marker='x', label='Nodes')
        
        # --- Оформление ---
        ax.set_title(f'Absolute error $|g(x) - L_{{{n-1}}}(x)|$ for $N={n}$', fontsize=14)
        ax.set_xlabel('$x$')
        ax.set_ylabel(f'$|g(x) - L_{{{n-1}}}(x)|$')
        ax.semilogy()  # Логарифмическая шкала по Y
        ax.grid(True, which="both", ls="-", alpha=0.5)
        # ax.legend()

    plt.tight_layout()
    plt.show()

# Пример вызова:



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
    plt.ylabel('max$|g(x)-L(x)|$')
    plt.grid()
    plt.semilogy()
    plt.legend()
    plt.show()

def plot_alphas(f, a, b, xc, n, alphas: list):
    """
    Строит 4 графика для ОДНОГО количества узлов n,
    но для РАЗНЫХ параметров сгущения alpha.
    """
    # Заранее считаем точную функцию для отрисовки фона
    # Берем побольше точек, чтобы видеть острые пики, если они будут
    ab = np.linspace(a, b, int((b - a) * 1000))
    f_val = f(ab)
    
    # Создаем сетку графиков 2x2
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    colors = ['red', 'orange', 'magenta', 'green']
    
    # Добавим общий заголовок, чтобы не забыть, какое у нас N
    fig.suptitle(f'Effect of parameter $\\alpha$ for $N={n}$', fontsize=16)

    # Итерируемся по массиву ALPHAS
    for i, alpha in enumerate(alphas):
        ax = axs[i]
        col = colors[i]
        
        # --- Вычисления ---
        # Теперь n фиксировано, а alpha меняется в каждой итерации
        x_h, y_h = table_function(f, a, b, xc, n, alpha)
        
        # Считаем полином
        p_val = np.array([Lagrange(x_h, y_h, x) for x in ab])
        
        # --- Рисование ---
        # 1. Точная функция g(x)
        ax.plot(ab, f_val, label='$g(x)$', color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 2. Полином Лагранжа
        ax.plot(ab, p_val, label=f'$L_{{{n-1}}}(x)$', color=col, linewidth=2)
        
        # 3. Узлы (рисуем их жирно, чтобы видеть, как они двигаются)
        ax.scatter(x_h, f(x_h), color=col, s=50, zorder=5, edgecolors='black')
        
        # --- Оформление ---
        ax.set_title(f'Density parameter $\\alpha = {alpha}$', fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(True)
        
        # Ограничение по Y (чтобы график не скакал при сильном Рунге)
        # Берем максимум функции, но даем запас на случай выбросов полинома
        limit = max(np.abs(f_val))
        # Если полином улетает очень далеко (Рунге), обрезаем его визуально, 
        # иначе полезная часть графика сожмется в линию.
        ax.set_ylim(-1.5 * limit, 2.0 * limit)

    plt.tight_layout()
    # Сдвигаем графики вниз, чтобы не перекрыть общий заголовок suptitle
    plt.subplots_adjust(top=0.92)
    plt.show()

# Пример вызова:
# Попробуй разные порядки alpha, чтобы увидеть разницу
# xc (центр сгущения) лучше ставить в корень функции (где излом)
# plot_alphas(g, -10, 10, xc=-1.8, n=40, alphas=[1e-5, 0.1, 1.0, 2.0])
def plot_1(f,a,b,xc,ns: list,alpha=1e-15):
    colors = ['red','orange','magenta','green','blue']
    if len(ns)<=5:
        for n in ns:
            col = colors[ns.index(n)]
            x_h, y_h = table_function(f,a,b,xc,n,alpha)
            ab = np.linspace(a,b,(b-a)*1000)
            p_val = np.array([Lagrange(x_h,y_h,x) for x in ab])
            plt.plot(ab,p_val,label = f'$L_{{{n-1}}}(x)$, {n} data points',color = col,linewidth = 2)
            plt.scatter(x_h,f(x_h),color = col)
        f_val = f(ab)
        plt.plot(ab,f_val,label = '$f(x)$',linewidth = 1.5,linestyle = 'dashed',color = 'black')
        plt.legend(fontsize = 16)
        plt.tick_params(labelsize = 16)
        plt.grid()
        plt.xlim(-7,5)
        plt.ylim(-1.1*max(f_val),1.5*max(f_val))

        
        plt.show()

a,b = -10,10
xc, al = -4.76994 , 0.05

n_list = [15]

# plot_1(f, a, b, 0, n_list, al)

# plot_2(f, a, b, 0, n_list, al)

# plot_3(f, a,b, xc,3,70, al)

# plot_alphas(g,a,b,xc,25,[0.06,0.064,0.067, 0.07])
plot_1(g,a,b,xc,[25],al)

# plot_1(g, a, b, xc, n_list, al)

# plot_2(g, a, b, xc, n_list, al)

# plot_3(g, a,b, xc,3,70, al)