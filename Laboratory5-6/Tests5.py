import numpy as np
import matplotlib.pyplot as plt
import time
from Leverier import get_spectrum


def matr_spectrum(spectrum):
    l = spectrum
    n = len(spectrum)
    d = np.diag(l)

    q = np.linalg.qr(np.random.rand(n, n)).Q
    a = np.matmul(np.matmul(q, d), q.T)
    return a


def test_time_dependency():
    print("--- ЗАПУСК ТЕСТА ПРОИЗВОДИТЕЛЬНОСТИ ---")
    dims = range(3, 31) 
    times = []

    for n in dims:

        true_spectrum = np.random.uniform(-10, 10, n)
        A = matr_spectrum(true_spectrum)
        
        start = time.time()
        try:

            _ = get_spectrum(A, 1e-6)
            elapsed = time.time() - start
        except:

            elapsed = 0
            
        times.append(elapsed)
        print(f"N={n}, Время: {elapsed:.4f} с")


    last_n = dims[-1]
    last_time = times[-1]
    theoretical_times = [last_time * ((n / last_n) ** 4) for n in dims]


    plt.figure(figsize=(10, 6))
    plt.plot(dims, times, 'o-', linewidth=2, label='Experimental data')
    # plt.plot(dims, theoretical_times, '--', color='orange', label='Theoretical estimate $O(n^4)$')


    plt.xlabel('dim(A)',fontsize = '18')
    plt.ylabel('Time, sec',fontsize = '18')
    plt.legend(fontsize = '18')
    plt.grid(True)
    plt.tick_params(labelsize = '16')

    plt.show()
test_time_dependency()


def test_accuracy_averaged(repeats=20):
    print(f"\n--- ЗАПУСК ТЕСТА ТОЧНОСТИ (Усреднение по {repeats} запускам) ---")
    
    dims = range(3, 31) 
    avg_errors = []
    valid_dims = []
    
    for n in dims:
        current_n_errors = []
        
        for _ in range(repeats):
            true_spectrum = np.sort(np.random.uniform(-10, 10, n))
            A = matr_spectrum(true_spectrum)
            
            try:
                calc_spectrum = get_spectrum(A, 1e-8)
                calc_spectrum = np.sort(np.array(calc_spectrum))
                
                if len(calc_spectrum) == n:
                    error = np.max(np.abs(calc_spectrum - true_spectrum))
                    current_n_errors.append(error)
            except:
                pass 
        

        if len(current_n_errors) > 0:
            mean_error = np.mean(current_n_errors)
            avg_errors.append(mean_error)
            valid_dims.append(n)
            print(f"N={n}, Средняя ошибка: {mean_error:.2e}")# (успешных тестов: {len(current_n_errors)}/{repeats})")
        else:
            print(f"N={n}, Все тесты провалились")


    plt.figure(figsize=(9, 6))
    

    plt.plot(valid_dims, avg_errors, 'o-', color='red', linewidth=2, label='Mean error')
    

    plt.yscale('log') 

    plt.xlabel('dim(A)',fontsize = '18')
    plt.ylabel('Mean absolute error (10 matrices)',fontsize = '18')
    plt.grid(True, which="both", ls="-", alpha=1)
    plt.tick_params(labelsize = '16')

    plt.legend()
    
    plt.tight_layout()

    plt.show()


# test_accuracy_averaged(repeats=10)