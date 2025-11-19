import time
import numpy as np
import matplotlib.pyplot as plt
from GEM import GEM_MOD 
from rotations import rotations

def analyze_rotations():
    sizes = []
    times_rotations = []
    times_gem = []
    
    print("Сравнение методов Гивенса и Гаусса:")
    print("n\tRotations\tGEM_MOD")
    print("-" * 40)
    
    for n in range(100, 1001, 100):
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        
        # Метод Гивенса
        start = time.time()
        rotations(A.copy(), b.copy())
        time_rot = time.time() - start
        
        # Метод Гаусса (ваш)
        start = time.time()
        GEM_MOD(A.copy(), b.copy())
        time_gem = time.time() - start
        
        sizes.append(n)
        times_rotations.append(time_rot)
        times_gem.append(time_gem)
        
        print(f"{n}\t{time_rot:.4f}\t\t{time_gem:.4f}")
    
    # Анализ сложности
    def get_exponent(sizes, times):
        log_n = np.log(sizes)
        log_t = np.log(times)
        return np.polyfit(log_n, log_t, 1)[0]
    
    exp_rot = get_exponent(sizes, times_rotations)
    exp_gem = get_exponent(sizes, times_gem)
    
    print(f"\nЭмпирическая сложность:")
    print(f"Метод Гивенса: O(n^{exp_rot:.2f})")
    print(f"Метод Гаусса: O(n^{exp_gem:.2f})")
    
    # График
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, times_rotations, 'ro-', label=f'Rotations (O(n^{exp_rot:.2f}))')
    plt.loglog(sizes, times_gem, 'bo-', label=f'GEM_MOD (O(n^{exp_gem:.2f}))')
    plt.xlabel('Размерность n')
    plt.ylabel('Время (сек)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Сравнение методов Гивенса и Гаусса')
    plt.show()

analyze_rotations()