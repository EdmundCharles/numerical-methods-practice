import numpy as np
from aux_functions import tr, matr_spectrum
def Leverier(A):
    n = len(A)
    p = np.zeros(n)
    first = [-1]
    s = np.zeros(n)
    for k in range(1,n+1):
        Ak = np.linalg.matrix_power(A,k)
        s[k-1] = tr(Ak)
        s_ = s[k-1]
        for i in range(1,k):
            s_ -= p[i-1]*s[k-i-1]
        p[k-1] = s_/k
    return np.concatenate((first,p))

def bisection_method(interval:tuple,eps,f,max_iter= 1000):
        a,b = interval[0],interval[1]
        c = (a+b)/2
        i = 0
        while abs(b-a)>2*eps and i < max_iter:
            i += 1
            c = (a+b)/2
            if f(a)*f(c)<0:
                b = c
            elif f(c) == 0:
                break
            else:
                a = c
        return float((a+b)/2)


def get_spectrum(A,eps):
    bound = np.linalg.norm(A,ord=np.inf)
    intervalish = (-bound,bound)
    spectrum = []
    coefs = Leverier(A)
    print(coefs)
    def polynomial(x):
        return sum(coefs[len(coefs)-1-i]*x**(i) for i in range(len(coefs)))
    p = polynomial
    print(p(0))
    a , b = intervalish[0],intervalish[1]
    toporik = []
    while a < b:
        c = a + eps
        if p(a)*p(c) <= 0:
            toporik.append((a,c))
        a += eps
    for interval in toporik:
        l = bisection_method(interval,1e-6,p)
        spectrum.append(l)
    return spectrum


A = matr_spectrum([1,2,3,4,5])
print(get_spectrum(A,1e-3))


# def intervals(interval,f):
#     a , b = interval[0],interval[1]
#     while True:
#         c = (a + b) /2
#         if f(a)*f(c) < 0:
