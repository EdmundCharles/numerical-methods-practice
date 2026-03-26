import numpy as np
import copy
def matr_spectrum(spectrum = None):
    l = spectrum
    n = len(spectrum)
    d = np.diag(l)
    q = np.linalg.qr(np.random.rand(n, n)).Q
    a = np.matmul(np.matmul(q, d), np.linalg.matrix_transpose(q))
    return a
def matr_sep(n,sep):
    l = 1
    step = abs(l - l*(1+sep))
    spectrum = np.zeros(n)
    spectrum[0] = l
    # spectrum[1] = l + l*sep
    for i in range(1,n):
        l += step
        spectrum[i] = l
    matrix = matr_spectrum(spectrum)
    return matrix
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

def perturb_matrix_random(A, epsilon):
    """Вносит случайное возмущение с уровнем epsilon"""
    E = np.random.randn(len(A),len(A))  
    delta_A = epsilon * np.linalg.norm(A) * (E / np.linalg.norm(E))
    Ac = copy.deepcopy(A)
    return (Ac + delta_A)





def getter(spectrum): 
    n = len(spectrum)
    A = matr_spectrum(spectrum)
    Q =  np.random.rand(n,n)
    Q_inv = np.linalg.inv(Q)
    B = Q@A@Q_inv
    # print('Q-----\n',Q,'\n')
    # print('Q_inverse\n',Q_inv,'\n')
    # print('B-----\n',B,'\n')
    return B

def get_mu(a):
    at = np.matrix_transpose(a)

    w = np.linalg.eig(a).eigenvectors[0]
    u = np.linalg.eig(at).eigenvectors[0]
    wnorm = np.linalg.norm(w)
    unorm = np.linalg.norm(u)
    dot = np.dot(w,u)
    mu = (wnorm*unorm)/dot
    return mu
# print(get_mu(B))

def get_matr_with_mu(spectrum,desired_mu):
    mu = 0
    i = 0
    while True:
        i += 1
        B = getter(spectrum)
        mu = get_mu(B)
        if abs(mu - desired_mu) < 100 and mu > desired_mu:
            break
    return B,mu


def get_intervals(a,b,f,power):
    def split(interval):
        k,m = interval[0],interval[1]
        mid = (k+m)/2
        bool1 = f(mid)*f(k)<=0
        print(bool1)
        bool2 = f(mid)*f(m)<=0
        print(bool2)
        if bool1 and bool2:
            return [[k,mid],[mid,m]]
        elif bool1:
            return [[k,mid],None]
        else:
            return [None,[mid,m]]
        
    intervals = [[a,b]]
    while True:
        intervals_new = []
        print(intervals)
        for i in intervals:
            if i != None:
                intervals_new += split(i)
        intervals = intervals_new 
        s = sum(1 for x in intervals if x is not None)
        print(s)
        if s == power:
            break
    return intervals

