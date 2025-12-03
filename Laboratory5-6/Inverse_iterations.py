import numpy as np
from helping_functions import matr_spectrum, LU_solve, LU_factorization



def Inverse_iterations(A,eps,max_iter = 1000):
    #Initial variables
    n = len(A)
    lu = LU_factorization(A)
    y = np.ones(n)
    l = 0
    counter = 1
    #Step 2: initial calculations
    s = np.dot(y,y)
    y_norm = np.sqrt(s)
    x = y/y_norm
    while counter < max_iter:
        #Step 3: solving Ay = x for next approximation
        counter +=1 
        y = LU_solve(lu,x)
        #Step 4: iterative calculations
        s = np.dot(y,y)
        t = np.dot(y,x)
        y_norm = np.sqrt(s)
        x = y/y_norm
        #Step 5: convergence test
        if abs(t/s - l) < eps:
            break
        l = t/s
    return l , x , counter


A = matr_spectrum(3,100,10000)
print(Inverse_iterations(A,1e-3))