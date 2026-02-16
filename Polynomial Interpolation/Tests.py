import numpy as np
import matplotlib.pyplot as plt

def generate_grid_demo():
    # --- 1. Параметры (как в вашей методичке) ---
    a, b = -1.0, 1.0   # Границы отрезка
    xc = 0.0           # Центр сгущения (x_c)
    alpha = 5      # Коэффициент сгущения (чем больше, тем сильнее сжатие)
    N = 11             # Количество узлов

    # --- 2. Математика (из вашего текста) ---
    
    # Границы в пространстве Z (через арктангенс)
    z_min = np.arctan(alpha * (a - xc))
    z_max = np.arctan(alpha * (b - xc))
    
    # Равномерная сетка в Z (шаг Delta z = const)
    z_nodes = np.linspace(z_min, z_max, N)
    
    # Обратное отображение в X (через тангенс)
    # Формула: x = xc + (1/alpha) * tan(z)
    x_nodes = xc + np.tan(z_nodes) / alpha

    # Гладкая кривая для отрисовки самого отображения (тангенсоиды)
    z_smooth = np.linspace(z_min, z_max, 200)
    x_smooth = xc + np.tan(z_smooth) / alpha

    # --- 3. Визуализация ---
    plt.figure(figsize=(10, 8))
    
    # Рисуем функцию отображения
    plt.plot(z_smooth, x_smooth, 'b-', lw=2, label=r'Отображение $x(z) = x_c + \frac{1}{\alpha}\tan(z)$')

    # Рисуем узлы и проекции
    for i in range(N):
        z_i = z_nodes[i]
        x_i = x_nodes[i]
        
        # Вертикальная линия от оси Z до кривой
        plt.plot([z_i, z_i], [a, x_i], 'k--', alpha=0.3)
        
        # Горизонтальная линия от кривой до оси X
        plt.plot([z_min, z_i], [x_i, x_i], 'r--', alpha=0.3)
        
        # Точка на кривой
        plt.plot(z_i, x_i, 'ko', markersize=4)

    # Отображаем узлы на осях
    # Ось Z (Равномерная)
    plt.plot(z_nodes, [a]*N, 'k|', markersize=10, markeredgewidth=2, label='Равномерная сетка $Z$')
    # Ось X (Сгущенная)
    plt.plot([z_min]*N, x_nodes, 'r_', markersize=10, markeredgewidth=2, label='Сгущенная сетка $X$')

    # --- Оформление ---
    plt.title(f'Сгущение к $x_c={xc}$ при $\\alpha={alpha}$', fontsize=16)
    plt.xlabel(r'Пространство $Z$', fontsize=16)
    plt.ylabel(r'Пространство $X$', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize = 16)
    
    # Ограничиваем оси для красоты
    plt.xlim(z_min - 0.1, z_max + 0.1)
    plt.ylim(a - 0.1, b + 0.1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_grid_demo()