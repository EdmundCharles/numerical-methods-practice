import numpy as np

def richardson_chebyshev(A, b, m=20, eps=1e-8, max_restarts=50):
    n = len(A)
    
    # Получаем параметры Чебышева
    alphas = get_alphas(A, m)
    
    x = np.zeros(n)  # Начальное приближение
    x_prev = x.copy()
    
    restart_count = 0
    while restart_count < max_restarts:
        # Сохраняем предыдущее решение для проверки сходимости
        x_prev_cycle = x.copy()
        
        # Выполняем m итераций Чебышева
        for k in range(m):
            residual = b - A @ x  # Правильная невязка
            alpha = alphas[k]
            x_next = x + alpha * residual
            
            x_prev, x = x, x_next
        
        # Проверяем сходимость по изменению решения
        diff_norm = np.linalg.norm(x - x_prev_cycle)
        residual_norm = np.linalg.norm(b - A @ x)
        
        print(f"Рестарт {restart_count + 1}: ||Δx|| = {diff_norm:.2e}, ||r|| = {residual_norm:.2e}")
        
        # Критерий остановки
        if residual_norm < eps:
            print(f"Сходимость достигнута за {restart_count + 1} циклов")
            break
            
        restart_count += 1
    
    return x

def get_alphas(A, m):
    """Корректное вычисление параметров Чебышева"""
    lowr, uppr = estimate_spectrum(A)
    
    # Проверяем корректность границ спектра
    if lowr <= 0:
        lowr = uppr * 0.01  # Защита от неположительных значений
    
    alphas = []
    
    # Вычисляем узлы Чебышева в правильном порядке
    for k in range(1, m + 1):
        theta = np.pi * (k - 0.5) / m
        t = np.cos(theta)
        lambda_k = (lowr + uppr)/2 + (uppr - lowr)/2 * t
        alphas.append(1.0 / lambda_k)
    
    # Можно применять в порядке убывания |alpha| для стабильности
    # alphas.sort(reverse=True, key=lambda x: abs(x))
    
    return alphas

def estimate_spectrum(A):
    """Улучшенная оценка спектра"""
    # Спектральный радиус через норму
    lambda_max = np.linalg.norm(A, 2)
    
    # Для положительно определенной матрицы
    lambda_min = 0.01 * lambda_max  # Консервативная оценка
    
    # Или можно использовать:
    # lambda_min = 1.0 / np.linalg.norm(np.linalg.pinv(A), 2)
    
    print(f"Оценка спектра: λ_min = {lambda_min:.2e}, λ_max = {lambda_max:.2e}")
    return lambda_min, lambda_max

# Создаем хорошо обусловленную тестовую задачу
def matr_cond(n,cond):
    """
    This function returns A -- a n by n matrix as a nested list.

    Parameters
    ----------
    n : int
        n = dim(A)
    cond : int,float
        desired condition number of A

    Returns
    -------
    A : array_like
        (n,n) nested list representing the matrix
    """
    l = np.linspace(1,cond,n)
    d = np.diag(l)
    q = np.linalg.qr(np.random.rand(n,n)).Q
    a = np.matmul(np.matmul(q,d),np.linalg.matrix_transpose(q))
    return a

A = matr_cond(1000,1e5)
x = np.ones(1000)
b = A @ x

solution = richardson_chebyshev(A, b, m=20, eps=1e-6,max_restarts= 1000000)
print(f"Финальная точность: {np.linalg.norm(b - A @ solution):.2e}")
print(f"Error {np.linalg.norm(x - solution)}")