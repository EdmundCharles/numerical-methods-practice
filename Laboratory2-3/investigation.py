import time
import numpy as np
import matplotlib.pyplot as plt
from GEM import GEM_Py
from rotations import rotations_py
from plotting import matr_cond, error

# matr = matr_cond(10,1e12)
# print(error(matr,GEM_Py))
# print(error(matr, rotations_py))

def test_internal_stability(n, cond):
    """Тест на внутреннюю устойчивость (к ошибкам округления)"""
    
    # Создаем плохо обусловленную матрицу
    A = matr_cond(n, cond)
    x_true = np.ones(n)
    b = A @ x_true
    
    # Решаем ОДНУ И ТУ ЖЕ систему обоими методами
    x_gem = GEM_Py(A, b)
    x_rot = rotations_py(A, b)
    
    # Ошибки относительно ИСТИННОГО решения
    error_gem = np.linalg.norm(x_gem - x_true) / np.linalg.norm(x_true)
    error_rot = np.linalg.norm(x_rot - x_true) / np.linalg.norm(x_true)
    
    return error_gem, error_rot

def plot_internal_stability():
    """График ошибок в зависимости от числа обусловленности"""
    conditions = np.logspace(10, 16, 20)
    errors_gem = []
    errors_rot = []
    
    for cond in conditions:
        eg, er = test_internal_stability(5, cond)
        errors_gem.append(eg)
        errors_rot.append(er)
    
    plt.loglog(conditions, errors_gem, 'o-', label='GEM')
    plt.loglog(conditions, errors_rot, 's-', label='Rotations')
    plt.xlabel('Число обусловленности κ(A)')
    plt.ylabel('Относительная ошибка')
    plt.legend()
    plt.grid(True)
    plt.show()
def test_extreme_conditions():

    conditions = [1e10, 1e12, 1e14, 1e16]
    
    for cond in conditions:
        print(f"\nκ(A) = {cond:.1e}")
        

        n = 5
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = 1.0 / (i + j + 1)
        

        U, s, Vt = np.linalg.svd(A)
        current_cond = s[0] / s[-1]
        scale = cond / current_cond
        A_scaled = A * np.sqrt(scale)
        

        s_scaled = np.linalg.svd(A_scaled, compute_uv=False)
        real_cond = s_scaled[0] / s_scaled[-1]
        print(f"Реальное κ(A): {real_cond:.2e}")
        
        x_true = np.ones(n)
        b = A_scaled @ x_true
        
        try:
            x_gem = GEM_Py(A_scaled.tolist(), b.tolist())
            x_rot = rotations_py(A_scaled.tolist(), b.tolist())
            
            error_gem = np.linalg.norm(x_gem - x_true) / np.linalg.norm(x_true)
            error_rot = np.linalg.norm(x_rot - x_true) / np.linalg.norm(x_true)
            
            print(f"GEM ошибка: {error_gem:.2e}")
            print(f"Вращения ошибка: {error_rot:.2e}")
            
            if error_rot > 0:
                ratio = error_gem / error_rot
                print(f"Отношение: {ratio:.2f}")
                
        except Exception as e:
            print(f"Ошибка: {e}")
# plot_internal_stability()

test_extreme_conditions()