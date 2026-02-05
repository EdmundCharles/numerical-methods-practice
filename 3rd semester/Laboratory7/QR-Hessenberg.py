import numpy as np
import time
from aux_functions import matr_spectrum

def rotation_coefficients(a, b):
    if b == 0:
        return 1.0, 0.0
    
    hypot = np.hypot(a, b)
    c = a / hypot
    s = b / hypot
    return c, s

def rotate(A, c, s, i, j):
    
    row_i = c * A[i, :] + s * A[j, :]
    row_j = -s * A[i, :] + c * A[j, :]
    A[i, :] = row_i
    A[j, :] = row_j

    # Работаем со столбцами (Right multiply by G)
    # G имеет столбцы: [c, -s] и [s, c]
    col_i = c * A[:, i] + s * A[:, j]
    col_j = -s * A[:, i] + c * A[:, j]
    A[:, i] = col_i
    A[:, j] = col_j

def to_hessenberg(A_in):
    """
    Приведение матрицы к форме Хессенберга методом вращений.
    Зануляем всё ниже первой поддиагонали.
    """
    A = A_in.copy()
    n = A.shape[0]
    
    # Идем по столбцам k
    for k in range(n - 2):
        # Идем по элементам под диагональю снизу вверх
        # (Нам нужно занулить A[i, k] используя A[i-1, k])
        for i in range(n - 1, k + 1, -1):
            if abs(A[i, k]) > 1e-10:
                # Вращаем плоскость (i-1, i)
                c, s = rotation_coefficients(A[i-1, k], A[i, k])
                rotate(A, c, s, i-1, i)
                
                # Принудительно ставим 0 для красоты (и точности)
                A[i, k] = 0.0
    return A

def qr_algorithm_eigenvalues(A_in, use_hessenberg=False, max_iter=1000, tol=1e-6):
    """
    Основной алгоритм поиска собственных чисел.
    """
    A = A_in.copy()
    n = A.shape[0]
    
    start_time = time.time()
    
    # 1. Этап Хессенберга (Опционально)
    if use_hessenberg:
        A = to_hessenberg(A)
        
    iterations = 0
    
    # 2. QR Итерации
    # Мы используем цикл, уменьшающий размерность матрицы ("Deflation"),
    # когда последний элемент под диагональю становится мал.
    m = n # Активный размер матрицы
    
    while m > 1 and iterations < max_iter:
        iterations += 1
        
        # Проверяем элемент под диагональю в последней строке активного блока
        # Если он мал, значит A[m-1, m-1] - это собственное число.
        if abs(A[m-1, m-2]) < tol:
            m -= 1
            continue

        # СДВИГ (Shift): используем A[m-1, m-1] (Rayleigh shift)
        # Это ускоряет сходимость.
        shift = A[m-1, m-1]
        
        # Вычитаем сдвиг (только на диагонали)
        idx = np.arange(m)
        A[idx, idx] -= shift
        
        # QR Шаг с помощью вращений
        # Если Hessenberg=True, нам нужно занулить ТОЛЬКО поддиагональ (O(N^2))
        # Если Hessenberg=False, по-честному бежим по всему низу (Naive O(N^3))
        
        limit_row = m if not use_hessenberg else min(m, 2) # Упрощенная логика для демо
        # В полной реализации Hessenberg бежит только i in range(m-1) по поддиагонали
        
        if use_hessenberg:
            # Оптимизированный проход: только вдоль диагонали
            for i in range(m - 1):
                if abs(A[i+1, i]) > 1e-12:
                    c, s = rotation_coefficients(A[i, i], A[i+1, i])
                    rotate(A, c, s, i, i+1)
        else:
            # Наивный проход: зануляем всё под диагональю (как при QR-разложении полной матрицы)
            # Идем по столбцам
            for j in range(m - 1):
                # Идем по строкам ниже диагонали
                for i in range(j + 1, m):
                     if abs(A[i, j]) > 1e-12:
                        c, s = rotation_coefficients(A[j, j], A[i, j]) # Используем диагональный элемент как опору
                        rotate(A, c, s, j, i)

        # Возвращаем сдвиг обратно
        A[idx, idx] += shift

    elapsed = time.time() - start_time
    return np.diag(A), iterations, elapsed

# --- ТЕСТИРОВАНИЕ ---

# Создаем случайную матрицу побольше
N = 30
np.random.seed(42)
# Делаем симметричную, чтобы С.Ч. были вещественными (для простоты сравнения)
A_test = matr_spectrum([1,2,3,4,5,6,7,8,9,10])

print(f"Матрица {N}x{N}")
print("-" * 40)

# 1. Запуск БЕЗ Хессенберга (Наивный)
vals_naive, iters_naive, time_naive = qr_algorithm_eigenvalues(A_test, use_hessenberg=False)
print(f"Naive QR:      {time_naive:.4f} сек | {iters_naive} итераций")
print(vals_naive)

# 2. Запуск С Хессенбергом
vals_hess, iters_hess, time_hess = qr_algorithm_eigenvalues(A_test, use_hessenberg=True)
print(f"Hessenberg QR: {time_hess:.4f} сек | {iters_hess} итераций")
print(vals_hess)
# 3. Проверка (NumPy)
vals_true = np.linalg.eigvals(A_test)
vals_true.sort()
# vals_hess.sort()

error = np.max(np.abs(vals_true - vals_hess))
print("-" * 40)
print(f"Макс. ошибка (vs numpy): {error:.2e}")
print(f"Ускорение: {time_naive / time_hess:.1f}x")