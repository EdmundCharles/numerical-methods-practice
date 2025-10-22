import matplotlib.pyplot as plt
import numpy as np
from Lab2 import GEM_MOD
import timeit
# def GEM_MOD_EXEC():

def matrix_sum(a,b):
    n = len(a)
    for i in range(n):
        for j in range(n):
            a[i][j] = a[i][j] + b[i][j]
    return a
def matrix_scalar_mult(a,c):
    n = len(a)
    for i in range(n):
        for j in range(n):
            a[i][j] = a[i][j]*c
    return a
###1.Condition number of matrix effect on the error

#Generating matrix A with a specified condition number
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
    return a.tolist()
def error(A,A_disturbed = None):
    """
    Calculates the error of the solution retuned by the GEM_MOD numerical method.
    If A_disturbed is specified(default is None), the function still uses A to calculate the x_num.
    
    Parameters
    ----------
    A : array_like
        (n,n) nested list representing the matrix A
    A_disturbed : array_like
        (n,n) nested list representing the disturbed matrix which is A+dA
    Returns
    -------
    error : int 
        The error calculated using numpy.linalg.norm() function 
    """
    
    n = len(A)
    x = [i for i in range(1,n+1)]
    b = np.matmul(A,x).tolist()
    if A_disturbed != None:
        x_num = GEM_MOD(A_disturbed,b)
    else:
        x_num = GEM_MOD(A,b)
    dx = [x[i]-x_num[i] for i in range(n)]
    error = np.linalg.norm(dx)/np.linalg.norm(x)
    return float(error)

def plot_err_cond(n):
    """
    Creates and shows a graph of the error as a function of condition number of a matrix. This method uses two other methods:
    1. matr_cond() to create a matrix with a specified condition number
    2. error() to calculate the error of the solution

    Parameters 
    ----------
    n : int
        The dimension of the matrix A
    """
    eps = 2.2e-16
    cond = np.logspace(1,9,1000)
    err = []
    max_errors = []
    for i in cond:
        a = matr_cond(n,i)
        err.append(error(a,A_disturbed=None))
        max_errors.append(i*(2*eps)/(1-eps*i))
    plt.scatter(cond,err,s=1.5)
    plt.plot(cond,max_errors,linestyle= 'dashed',color = 'red',label = 'Теоретическая граница для погрешности')
    plt.xlabel('$cond(A)$')
    plt.ylabel('$\\frac{||\\delta x||}{||x||}$',loc='center',rotation= 'horizontal',size='14',labelpad=20)
    plt.loglog()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def perturb_matrix_random(A, epsilon):
    """Вносит случайное возмущение с уровнем epsilon"""
    E = np.random.randn(len(A),len(A))  # Случайная матрица
    delta_A = epsilon * np.linalg.norm(A) * (E / np.linalg.norm(E))
    return (A + delta_A).tolist()
def plot_err_deltaA(n,cond):
    """
    
    """

    epsilons = np.logspace(-9,-2,100).tolist()
    a = matr_cond(n,cond)
    errors = []
    max_errors = []
    for eps in epsilons:
        a_eps = perturb_matrix_random(a,eps)
        errors.append(error(a,a_eps))
        max_errors.append(cond*eps)
    
    plt.figure(figsize=(10,6))
    plt.scatter(epsilons,errors,s=3,label = 'Experimental data')
    plt.plot(epsilons,max_errors,linestyle = 'dashed',label='$\\frac{||\\delta x||}{||x||}=cond(A) \\cdot \\frac{||\\delta A||}{||A||}$',color = 'red')
    plt.xlabel('$\\frac{||\\delta A||}{||A||}$',size = 16)
    plt.ylabel('$\\frac{||\\delta x||}{||x||}$',loc='center',rotation= 'horizontal',labelpad=20,size = 16)
    plt.loglog()
    plt.legend()
    plt.grid(True,alpha = 0.3)
    plt.tight_layout()
    plt.show()


def plot_comp_n(cond):
    times = []
    dims = []
    for n in range(10,1000,10):
        dims.append(n)
        a = matr_cond(n,cond)
        b = [i for i in range(n)]
        time = timeit.timeit(lambda: GEM_MOD(a,b),number=1)
        times.append(time)
        print(n,time)
    dims3 = [i**3 for i in dims]
    alphas = [times[i]/dims3[i] for i in range(len(times))]
    alpha = np.mean(alphas)
    print(alpha)
    times_theor = [alpha*i for i in dims3]

    plt.figure(figsize=(10,6))
    plt.plot(dims,times,label = 'Experimental data')
    plt.plot(dims,times_theor,label = '$t = \\alpha \cdot n^3$')
    plt.xlabel('$n = dim(A)$')
    plt.ylabel('Execution time (t), c')
    plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    plt.legend()
    plt.loglog()
    plt.show()
    #

plot_err_cond(10)
plot_err_cond(100)
plot_err_deltaA(100,100)
plot_err_deltaA(100,1e7)
plot_comp_n(100)