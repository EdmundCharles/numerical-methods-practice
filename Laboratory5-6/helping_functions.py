import numpy as np

def matr_spectrum(n,lmin,lmax):
    l = np.linspace(lmin, lmax, n)
    d = np.diag(l)
    q = np.linalg.qr(np.random.rand(n, n)).Q
    a = np.matmul(np.matmul(q, d), np.linalg.matrix_transpose(q))
    return a

def LU_factorization(A):
    n = len(A)
    lu = np.zeros((n,n))
    for i in range(n):
        
        for k in range(i,n):
            s = 0
            for j in range(i):
                s += lu[i,j]*lu[j,k]
            lu[i,k] = A[i,k] - s
        
        for k in range(i+1,n):
                
            s = 0
            for j in range(i):
                    s += lu[k,j]*lu[j,i]
            lu[k,i] = (A[k,i] - s)/lu[i,i]
    return lu

def LU_solve(lu,b):
    n = len(lu)
    y = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        s = 0
        for k in range(i):
            s += lu[i,k]*y[k]
        y[i] = b[i] - s
    for i in range(n-1,-1,-1):
        s = 0 
        for k in range(i+1,n):
            s +=lu[i,k]*x[k]
        x[i] = 1/lu[i,i]*(y[i]-s)

    return x

def tr(A):
    tr = 0
    for i in range(len(A)):
        tr += A[i,i]
    return tr

