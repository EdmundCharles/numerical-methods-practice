import numpy as np
import numpy.linalg as la

A = np.array([[-3-1j,-5,11],[-4,-8-1j,17],[-2,-5,10-1j]])
# print(la.eig(A).eigenvectors)
u = la.solve(A,np.zeros(3))

import numpy as np

def null_space(A, atol=1e-13, rtol=0):
    """
    Находит ядро (нуль-пространство) матрицы A.
    
    Параметры:
    A : numpy array, размер (m, n)
    atol : абсолютный допуск для нулевых сингулярных значений
    rtol : относительный допуск для нулевых сингулярных значений
    
    Возвращает:
    Z : ортонормированный базис ядра (столбцы образуют базис ядра)
    """
    A = np.atleast_2d(A)
    m, n = A.shape
    
    # Сингулярное разложение
    U, s, Vh = np.linalg.svd(A)
    
    # Определение допуска для нулевых сингулярных значений
    tol = max(atol, rtol * s[0])
    
    # Количество нулевых сингулярных значений
    num_zero = np.sum(s <= tol)
    
    # Базис ядра - это последние num_zero строк Vh
    Z = Vh[-num_zero:].T
    
    return Z

print(A @ np.array([5,8-1j,5]))
