import numpy as np
import matplotlib.pyplot as plt

def plot_err_deltaA(n, cond, num_points=50, use_worst_case=False):
    """
    Исследует влияние возмущения матрицы A на погрешность решения
    
    Parameters:
    -----------
    n : int
        Размер матрицы
    cond : float
        Желаемое число обусловленности
    num_points : int
        Количество точек для исследования
    use_worst_case : bool
        Если True, использует наихудшее возмущение через SVD
        Если False, использует случайные возмущения
    """
    
    # Генерируем матрицу с заданным числом обусловленности
    A = generate_matrix_with_condition(n, cond)
    
    # Вычисляем норму матрицы A
    norm_A = np.linalg.norm(A, 2)
    
    # Создаем точное решение (вектор из единиц)
    x_true = np.ones(n)
    
    # Вычисляем правую часть
    b = A @ x_true
    
    # Диапазон возмущений
    epsilons = np.logspace(-8, -2, num_points)
    
    errors = []
    
    for eps in epsilons:
        if use_worst_case:
            # Наихудшее возмущение через SVD
            delta_A = generate_worst_case_perturbation(A, eps, norm_A)
        else:
            # Случайное возмущение
            delta_A = generate_random_perturbation(A, eps, norm_A)
        
        # Возмущенная матрица
        A_perturbed = A + delta_A
        
        try:
            # Решаем возмущенную систему
            x_perturbed = np.linalg.solve(A_perturbed, b)
            
            # Вычисляем относительную погрешность
            rel_error = np.linalg.norm(x_perturbed - x_true, 2) / np.linalg.norm(x_true, 2)
            errors.append(rel_error)
        except np.linalg.LinAlgError:
            # Если матрица стала вырожденной
            errors.append(np.nan)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, errors, 'o-', markersize=4, linewidth=1, label='Эксперимент')
    
    # Теоретическая оценка (верхняя граница)
    theoretical_bound = cond * epsilons / (1 - cond * epsilons)
    plt.plot(epsilons, theoretical_bound, 'r--', linewidth=2, 
             label=f'Теоретическая граница (cond={cond})')
    
    plt.xlabel('$\\varepsilon = \\frac{\\|\\delta A\\|}{\\|A\\|}$', fontsize=12)
    plt.ylabel('$\\frac{\\|\\delta x\\|}{\\|x\\|}$', fontsize=12, rotation=0, labelpad=20)
    plt.title(f'Влияние возмущения матрицы на погрешность решения\n'
              f'n={n}, cond(A)={cond}')
    plt.loglog()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return epsilons, errors

def generate_matrix_with_condition(n, cond):
    """Генерирует матрицу с заданным числом обусловленности"""
    # Создаем случайную ортогональную матрицу U
    U, _ = np.linalg.qr(np.random.randn(n, n))
    # Создаем случайную ортогональную матрицу V
    V, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Создаем диагональную матрицу с сингулярными числами
    sigma_max = 1.0
    sigma_min = sigma_max / cond
    sigma_values = np.linspace(sigma_max, sigma_min, n)
    Sigma = np.diag(sigma_values)
    
    # Собираем матрицу A = U Σ Vᵀ
    A = U @ Sigma @ V.T
    return A

def generate_random_perturbation(A, eps, norm_A):
    """Генерирует случайное возмущение"""
    n = A.shape[0]
    # Случайная матрица с нормальным распределением
    E = np.random.randn(n, n)
    # Нормализуем и масштабируем
    norm_E = np.linalg.norm(E, 2)
    delta_A = (eps * norm_A / norm_E) * E
    return delta_A

def generate_worst_case_perturbation(A, eps, norm_A):
    """Генерирует наихудшее возмущение через SVD"""
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    
    # Берем сингулярные векторы, соответствующие min сингулярному числу
    u_min = U[:, -1].reshape(-1, 1)
    v_min = V[:, -1].reshape(-1, 1)
    
    # Наихудшее возмущение: внешнее произведение
    delta_A = eps * norm_A * (u_min @ v_min.T)
    return delta_A

def error(A, A_perturbed):
    """Вычисляет погрешность решения (альтернативная реализация)"""
    n = A.shape[0]
    x_true = np.ones(n)
    b = A @ x_true
    
    try:
        x_perturbed = np.linalg.solve(A_perturbed, b)
        rel_error = np.linalg.norm(x_perturbed - x_true, 2) / np.linalg.norm(x_true, 2)
        return rel_error
    except np.linalg.LinAlgError:
        return np.nan
    
# Исследование со случайными возмущениями
epsilons, errors = plot_err_deltaA(n=10, cond=1000, use_worst_case=False)

# Исследование с наихудшими возмущениями
# epsilons, errors = plot_err_deltaA(n=10, cond=1000, use_worst_case=True)