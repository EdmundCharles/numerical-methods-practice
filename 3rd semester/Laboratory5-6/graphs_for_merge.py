import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def power_method(A: np.ndarray, x0: np.ndarray, epsilon: float, max_iter: int = 10000) -> Tuple[float, np.ndarray, int]:
    """
    Степенной метод для нахождения максимального по модулю собственного числа
    """
    x = x0 / np.linalg.norm(x0)
    
    for i in range(max_iter):
        y = A @ x
        lambda_approx = np.dot(x, y) / np.dot(x, x)
        x_new = y / np.linalg.norm(y)
        
        if np.linalg.norm(x_new - x) < epsilon:
            return lambda_approx, x_new, i + 1
        
        x = x_new
    
    return lambda_approx, x, max_iter

def create_test_matrix(eigenvalues: List[float]) -> np.ndarray:
    """
    Создает матрицу с заданными собственными числами
    """
    n = len(eigenvalues)
    D = np.diag(eigenvalues)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    A = Q @ D @ Q.T
    return A

def test_convergence(eigenvalues: List[float], epsilons: List[float]) -> List[int]:
    """
    Тестирует сходимость для одной матрицы и различных точностей
    """
    A = create_test_matrix(eigenvalues)
    n = A.shape[0]
    iterations_list = []
    
    for epsilon in epsilons:
        x0 = np.random.randn(n)
        _, _, iterations = power_method(A, x0, epsilon)
        iterations_list.append(iterations)
    
    return iterations_list

# Параметры тестирования
epsilons = np.logspace(-10,-2,100)

# Тестирование матриц
print("Тестирование сходимости...")

# Матрица с хорошей отделимостью
good_eigenvalues = [10.0, 1.0, 0.5, 0.2]  # |λ₂/λ₁| = 0.1
good_iterations = test_convergence(good_eigenvalues, epsilons)

# Матрица с плохой отделимостью  
poor_eigenvalues = [10.0, 9.9, 2.0, 1.0]  # |λ₂/λ₁| = 0.95
poor_iterations = test_convergence(poor_eigenvalues, epsilons)

# Построение графиков на одном чертеже
plt.figure(figsize=(12, 8))

# Оба графика на одном чертеже
plt.plot(epsilons, good_iterations, 'bo-', linewidth=2, markersize=6, label='Хорошая отделимость')
plt.plot(epsilons, poor_iterations, 'ro-', linewidth=2, markersize=6, label='Плохая отделимость')

plt.xscale('log')
plt.yscale('log')  # Добавлена логарифмическая шкала по Y
plt.xlabel('Точность ε', fontsize=12)
plt.ylabel('Число итераций', fontsize=12)
# plt.title('Сравнение сходимости степенного метода', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Убедимся, что оси идут от меньшего к большему
plt.gca().set_xlim(1e-10, 1e-2)
plt.gca().set_ylim(1, max(max(good_iterations), max(poor_iterations)) * 1.5)  # Изменены пределы для log scale

plt.tight_layout()
plt.show()

# Вывод результатов в таблицу
print("\nРезультаты тестирования:")
print("Точность ε\tХорошая отдел.\tПлохая отдел.")
print("-" * 45)
for i, eps in enumerate(epsilons):
    print(f"{eps:.0e}\t\t{good_iterations[i]}\t\t{poor_iterations[i]}")

print(f"\nОтношение |λ₂/λ₁| для хорошей отделимости: {abs(good_eigenvalues[1]/good_eigenvalues[0]):.3f}")
print(f"Отношение |λ₂/λ₁| для плохой отделимости: {abs(poor_eigenvalues[1]/poor_eigenvalues[0]):.3f}")