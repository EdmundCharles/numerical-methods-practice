import numpy as np
import matplotlib.pyplot as plt
from Cubic_spline import cubic_spline
# Если запускаете в отдельном файле, раскомментируйте строку ниже:
# from aux_func import cubic_spline

# 1. Генерируем тестовые данные (узлы интерполяции)
# Возьмем 6 точек на отрезке от 0 до 2*pi
x_nodes = np.linspace(0, 2 * np.pi, 6)
y_nodes = np.sin(x_nodes)

# 2. Вычисляем сплайн с помощью вашей функции
# Передаем 0 в качестве краевых условий (sd_a=0, sd_b=0), 
# чтобы получить классический "естественный" кубический сплайн
spline_vals, ab = cubic_spline(x_nodes, y_nodes, sd_a=1e-3, sd_b=1e-3)

# 3. Для наглядности сгенерируем "идеальный" синус с высокой точностью
x_ideal = np.linspace(0, 2 * np.pi, 100)
y_ideal = np.sin(x_ideal)

# 4. Строим график
plt.figure(figsize=(10, 6))

# Идеальная функция (полупрозрачная серая линия)
plt.plot(x_ideal, y_ideal, color='gray', linestyle='--', alpha=0.7, label='Истинная функция: sin(x)')

# Ваш сплайн (синяя линия)
plt.plot(ab, spline_vals, color='blue', linewidth=2, label='Ваш кубический сплайн')

# Исходные узлы (красные точки)
plt.scatter(x_nodes, y_nodes, color='red', s=50, zorder=5, label='Узлы интерполяции')

# Наводим красоту
plt.title('Тестирование кубического сплайна', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12)

# Показываем результат
plt.show()


import numpy as np
import matplotlib.pyplot as plt
# from aux_func import cubic_spline # раскомментируйте, если запускаете в другом файле

# Функция Рунге
def runge(x):
    return 1 / (1 + 25 * x**2)

# 1. Берем 11 узлов на отрезке [-1, 1]
n_points = 11
x_nodes = np.linspace(-1, 1, n_points)
y_nodes = runge(x_nodes)

# 2. Строим ваш кубический сплайн
# Задаем краевые условия равными нулю (свободные концы)
spline_vals, ab = cubic_spline(x_nodes, y_nodes, sd_a=1e-3, sd_b=1e-3)

# 3. Для сравнения строим классический интерполяционный многочлен (10-й степени для 11 точек)
poly_coeffs = np.polyfit(x_nodes, y_nodes, n_points - 1)

# 4. Идеальная кривая для отрисовки
x_ideal = np.linspace(-1, 1, 500)
y_ideal = runge(x_ideal)
y_poly = np.polyval(poly_coeffs, x_ideal)

# 5. Рисуем графики
plt.figure(figsize=(12, 7))

# Обычный многочлен (ужас на краях)
plt.plot(x_ideal, y_poly, color='orange', linestyle='--', linewidth=2, 
         label=f'Полином {n_points-1}-й степени (Феномен Рунге)')

# Идеальная функция
plt.plot(x_ideal, y_ideal, color='gray', linewidth=4, alpha=0.4, 
         label='Истинная функция Рунге')

# ВАШ кубический сплайн
plt.plot(ab, spline_vals, color='blue', linewidth=2, 
         label='Ваш кубический сплайн')

# Узлы
plt.scatter(x_nodes, y_nodes, color='red', s=60, zorder=5, label='Узлы интерполяции')

# Ограничим ось Y, иначе полином улетит так высоко, что сплайна не будет видно
plt.ylim(-0.5, 1.5)

plt.title('Битва алгоритмов: Кубический сплайн против Феномена Рунге', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, loc='upper center')

plt.show()