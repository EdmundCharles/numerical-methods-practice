import matplotlib.pyplot as plt
import numpy as np


def matr_cond(n, cond):
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
    l = np.linspace(1, cond, n)
    d = np.diag(l)
    q = np.linalg.qr(np.random.rand(n, n)).Q
    a = np.matmul(np.matmul(q, d), np.linalg.matrix_transpose(q))
    return a


def get_alphas(A, m, lmin, lmax, sort=False):
    lowr, uppr = lmin, lmax
    alphas = [0]*m
    for k in range(1, m+1):
        t = np.cos(np.pi*(2*k-1)/(2*m))
        l = (lowr+uppr)/2 + (uppr-lowr)/2*t
        alphas[k-1] = 1/l
    if sort:
        print('Sorting....')
        alphas.sort(reverse=True, key=lambda x: abs(x))
    return alphas


def saver(method: callable, param):
    name = method.__name__
    print(name)
    print('called')
    full = name + param
    plt.savefig(r'c:\Users\ilyad\Downloads\\'+f'{full}'+'.pdf')


def err_iter(n, cond, m=10, eps=1e-7, max_iter=1e4):
    A = matr_cond(n, cond)
    x_star = np.ones(n)
    b = A @ x_star
    alphas = get_alphas(A, m, 1, cond)
    x = np.zeros(n)
    count = 0
    errors = []
    iters = []
    while count < max_iter:
        for k in range(m):
            alpha = alphas[k]
            residual = A @ x - b
            x_next = x - alpha*residual
            x = x_next

            residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
            err = np.linalg.norm(x-x_star)/np.linalg.norm(x_star)
            errors.append(err)
            count += 1
            iters.append(count)
        if residual_norm*cond < eps:
            print(f'Требуемая точность достигнута за {count} серий')
            break
        print(f"Restart {count + 1}: ||r|| = {residual_norm:.2e}")

    plt.plot(iters, errors, color='red')
    plt.ylabel('$\\frac{||\\delta x||}{||x||}$', loc='center',
               rotation='horizontal')
    plt.xlabel('Series ran')

    plt.loglog()
    plt.tight_layout()
    plt.grid()
    saver(err_iter, param='_cond='+str(float(np.log10(cond))))
    plt.show()


def i_cond(n, eps=1e-5, m=10, max_iter=1e4):
    conds = np.logspace(1, 5, 10)
    iters = []
    for cond in conds:
        a = matr_cond(1000, cond)
        alphas = get_alphas(a, m, 1, cond)
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
                err_approx = residual_norm*cond

            if err_approx < eps:
                print(f'Требуемая точность достигнута за {count} серий')
                break
            print(f"Рестарт {count + 1}: ||r|| = {err_approx:.2e}")
        iters.append(count)
    plt.plot(conds, iters, color='red')
    plt.ylabel('Series required', loc='center',
               rotation='vertical', size='14', labelpad=20)
    plt.xlabel('Condition number')
    plt.loglog()
    plt.tight_layout()
    plt.grid()
    plt.show()


err_iter(100,1e2,m=10)
i_cond(1000)
# def iter_m():
#     ms = np.linspace(1, 100, 100)
#     iters = []
#     for m in ms:

