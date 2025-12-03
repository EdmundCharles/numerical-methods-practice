import numpy as np
def estimate_spectrum(A):
    m = np.min(np.linalg.eigvals(A))
    M = np.max(np.linalg.eigvals(A))
    return m, M
def get_alphas(A,m):
    lowr , uppr = estimate_spectrum(A) 
    alphas = [0]*m
    for k in range(1,m+1):
        t = np.cos(np.pi*(2*k-1)/(2*m))
        l = (lowr+uppr)/2 + (uppr-lowr)/2*t
        alphas[k-1] = 1/l
    # alphas.sort(reverse = True, key= lambda x: abs(x))
    alphas = sorter(alphas)
    return alphas
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


def richardson(A,b,m=10,eps = 1e-7,max_iter = 1e4):
    n = len(A)
    cond = np.linalg.cond(A)
    alphas = get_alphas(A,m)
    x = np.zeros(n)
    count = 0
    while count < max_iter:
        count +=1
        for k in range(m):
            alpha = alphas[k]
            residual = A @ x - b
            x_next = x - alpha*residual
            x = x_next

            residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
            
        if residual_norm*cond < eps:
            print(f'Требуемая точность достигнута за {count} серий')
            break
        print(f"Рестарт {count + 1}: ||r|| = {residual_norm:.2e}")
    return x


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

a = matr_cond(100,10)
x = np.ones(100)
b = a @ x
x_num = richardson(a,b,m=256,max_iter=1e6,eps=1e-6)
print('Absolute error=',np.linalg.norm(x-x_num))
print('Relative Error=', np.linalg.norm(x - x_num)/np.linalg.norm(x))
