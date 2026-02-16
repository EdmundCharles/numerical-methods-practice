import numpy as np
import matplotlib.pyplot as plt

def get_x_grid(a, b, xc, N, alpha=1.0):
    """
    Генерирует гладкую сетку с использованием жуткого арктангенса.
    
    Параметры:
    alpha : коэффициент сгущения. 
            0 -> равномерная сетка (в пределе)
            > 0 -> сгущение к xc
            Чем больше alpha, тем сильнее концентрация.
    """
    y_min = np.arctan(alpha * (a - xc))
    y_max = np.arctan(alpha * (b - xc))
    
    y_uniform = np.linspace(y_min, y_max, N)
    
    x_grid = xc + np.tan(y_uniform) / alpha
    
    return x_grid

def table_function(f,a,b,xc,n,alpha = 0.1):
    x_h = get_x_grid(a,b,xc,n,alpha)
    y_h = f(x_h)
    return x_h,y_h




# # --- Параметры ---
# a, b = 0, 100
# xc =   6 # Точка концентрации
# N = 15
# alpha = 0.1# Попробуйте 0.5 (слабо) или 5.0 (сильно)

# # Генерация
# nodes = create_arctan_grid(a, b, xc, N, alpha)

# # --- Визуализация ---
# plt.figure(figsize=(10, 6))

# # Рисуем сами узлы на оси
# plt.scatter(nodes, np.zeros_like(nodes), color='red', s=20, label='Узлы сетки', zorder=5)

# # График шага сетки (расстояние между соседями)
# h = np.diff(nodes)
# x_centers = (nodes[:-1] + nodes[1:]) / 2
# plt.plot(x_centers, h, 'b-', label='Шаг сетки h(x)')

# plt.axvline(xc, color='gray', linestyle='--', alpha=0.5, label='Центр (xc)')
# plt.title(f'Сетка методом Арктангенса (alpha={alpha})')
# plt.xlabel('x')
# plt.ylabel('Размер шага h')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# # Проверка границ (важно для численных методов)
# print(f"Первый узел: {nodes[0]:.4f} (должен быть {a})")
# print(f"Последний узел: {nodes[-1]:.4f} (должен быть {b})")