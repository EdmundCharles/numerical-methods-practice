import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(1,1e7)
# f = lambda x: 10*np.sqrt(x)
# plt.plot(x,f(x))
# plt.loglog()
# plt.grid()
# plt.show()
def sorter(coefficients):
    n = len(coefficients)
    if (n & (n - 1)) != 0 or n < 1:
        raise ValueError("n must be a power of 2")
    
    if n == 1:
        return coefficients
    
    permutation = [1, 2]
    current_size = 2
    
    while current_size < n:
        next_size = current_size * 2
        new_permutation = []
        for j in permutation:
            new_permutation.append(j)
            new_permutation.append(next_size + 1 - j)
        permutation = new_permutation
        current_size = next_size
    
    return [coefficients[j-1] for j in permutation]
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
def get_alphas(m, lmin, lmax, sort):
    lowr, uppr = lmin, lmax
    alphas = [0]*m
    for k in range(1, m+1):
        t = np.cos(np.pi*(2*k-1)/(2*m))
        l = (lowr+uppr)/2 + (uppr-lowr)/2*t
        alphas[k-1] = 1/l
    print(len(alphas))
    if sort:
        # alphas.sort(reverse=True, key=lambda x: abs(x))
        alphas = sorter(alphas)
    return alphas
def i_cond(n, eps=1e-3, m=10, max_iter=1e4):
    conds = np.logspace(1,5, 10)
    iters = []
    for cond in conds:
        a = matr_cond(n, cond)
        alphas = get_alphas(m, 1, cond, True)
        x_star = np.ones(n)
        b = a @ x_star
        x = np.zeros(n)
        count = 0
        while count < max_iter:
            count += 1
            for k in range(m):
                alpha = alphas[k]
                residual = a @ x - b
                x_next = x - alpha*residual
                x = x_next

                residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
                err_approx = residual_norm

            if err_approx < eps:
                print(f'Требуемая точность достигнута за {count} серий')
                break
            print(f"Рестарт {count + 1}: ||r|| = {err_approx:.2e}")
        iters.append(count*m)

    return conds,iters

x,y = i_cond(100,m=128)

def power_fit(x, y):
    """
    Аппроксимирует данные степенной зависимостью y = a * x^b
    
    Parameters:
    x, y: массивы данных
    
    Returns:
    a, b: параметры аппроксимации
    r_squared: коэффициент детерминации
    """
    # Логарифмируем данные для линейной регрессии
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Линейная регрессия в логарифмических координатах
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)
    
    # Вычисляем R²
    y_pred = a * x**b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared

print(power_fit(x,y))