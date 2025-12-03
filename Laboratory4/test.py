import numpy as np
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
def get_alphas(A,m,lowr,uppr):
    print(f'Спектр матрицы А \lambda_min = \delta = {lowr}, \lambda_max = \Delta = {uppr}')
    alphas = [0]*m
    for k in range(1,m+1):
        print(f'k = {k}')
        print()
        t = np.cos(np.pi*(2*k-1)/(2*m))
        print(f't_{k} = cos(\pi\cdot({2*k-1})/{2*m}) = {t}')
        l = (lowr+uppr)/2 + (uppr-lowr)/2*t
        print(f'\lambda_{k} = {(lowr + uppr)/2} + {(uppr- lowr)/2}\cdot {t}  = {l}')
        alphas[k-1] = 1/l
        print(f'\\alpha_{k} = 1/{l} = {alphas[k-1]}')
    print(f'В итоге получено \n', *alphas)
    return alphas


def richardson_demo(cond = 10, n = 3,m=10,eps = 1e-7,max_iter = 1e4):
    A = matr_cond(n,cond)
    print(f'Матрица А = \n {A}')
    x_star = np.ones(n)
    print(f'x^* = {x_star}')
    b = A @ x_star
    print(f'Тогда вектор b = {b}')
    
    alphas = get_alphas(A,m,1,cond)
    
    x = np.zeros(n)
    print(f'Начальное приближение : x^({{0}}) = {x}')
    count = 0
    while count < max_iter:
        count +=1
        print(f'Cерия {count}')
        print()
        for k in range(m):
            alpha = alphas[k]
            print(f'\\alpha = {alpha}')
            residual = A @ x - b
            print(f'r = {residual}')
            residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
            print(f'||r||_{{отн}} = ||r||/||b|| = {residual_norm}')
            x_next = x - alpha*residual
            print(f'x^({k+1}) = x^({k}) - \\alpha_{k}(Ax^({k})-b) = {x_next}')
            x = x_next    
        if residual_norm*cond < eps:
            print(f'||r||_{{отн}}*cond(A) = {residual_norm*cond} < {eps}')
            print(f'Требуемая точность достигнута за {count} серий')
            break
        print(f'||r||_{{отн}}*cond(A) = {residual_norm*cond} > {eps}')
        print('Точность не достигнута, продолжаем итерации…')
    
    print(f'Относительная погрешность решения ||x^{{*}} - x||/||x^{{*}}|| = {np.linalg.norm(x - x_star)/np.linalg.norm(x_star)}')


richardson_demo(m = 4, eps = 1e-3)