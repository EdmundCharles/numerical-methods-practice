import numpy as np
from aux_functions import tr, matr_spectrum


def Leverier_legacy(A):
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


def Leverier(A):
    n = len(A)
    p = np.zeros(n)
    first = np.array([-1]) 
    s = np.zeros(n)
    
    B = np.eye(n)
    
    for k in range(1, n + 1):

        B = np.dot(B, A) 
        
        s[k-1] = np.trace(B) 
        
        s_ = s[k-1]
        for i in range(1, k):
            s_ -= p[i-1] * s[k-i-1]
        p[k-1] = s_ / k
        
    return np.concatenate((first, p))
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


def get_derivative_coefs(coefs):
    """Возвращает коэффициенты производной полинома."""
    n = len(coefs) - 1
    deriv = []
    for i in range(n):
        deriv.append(coefs[i] * (n - i))
    return np.array(deriv)


def find_real_roots_recursive(coefs, bound, eps):
    """Используется теорема Ролля (почти очевидная) для корней многочлена.
    Корни самого многочлена лежат между корнями производной, что почти понятно, в случае простых корней, еще и единственность гарантирована. 
    Сам алгоритм рекурсивно доходит до линейного уравнения (последовательно дифференцируя) и решает его, а дальше обратно вверх идет, 
    находя последовательно корни производных все меньшего порядка"""
    def polynomial(x):
        return sum(coefs[len(coefs)-1-i]*x**(i) for i in range(len(coefs)))
    f = polynomial
    #max recursion depth (ax+b=0)
    if len(coefs) == 2:

        return [-coefs[1] / coefs[0]]
    

    deriv_coefs = get_derivative_coefs(coefs)
    
    extremums = find_real_roots_recursive(deriv_coefs, bound, eps)
    extremums.sort()
    

    current_points = [-bound] + extremums + [bound]
    
    roots = []

    for i in range(len(current_points) - 1):
        a, b = current_points[i], current_points[i+1]
        
        root = bisection_method((a, b), eps,f)
        
        if root is not None:
            roots.append(root)
                
    return roots

def get_spectrum_legacy(A,eps):
    bound = np.linalg.norm(A,ord=np.inf)
    intervalish = (-bound,bound)
    spectrum = []
    coefs = Leverier(A)
    def polynomial(x):
        return sum(coefs[len(coefs)-1-i]*x**(i) for i in range(len(coefs)))
    p = polynomial
    a , b = intervalish[0],intervalish[1]
    toporik = []
    while a < b:
        c = a + eps
        if p(a)*p(c) <= 0:
            toporik.append((a,c))
        a += eps
    for interval in toporik:
        l = bisection_method(interval,1e-9,p)
        spectrum.append(l)
    
    # return np.array(spectrum)
    return spectrum

def get_spectrum(A,eps):
    bound = np.linalg.norm(A,ord=np.inf)*1.1
    
    coefs = Leverier(A)

    spectrum = find_real_roots_recursive(coefs,bound,eps)
    return spectrum

