import matplotlib.pyplot as plt
import numpy as np
from aux_functions import LU_solve, LU_factorization, matr_sep

#The modified INVIT function returning more data, which is going to be used in the following tests.
def INVIT_demo(A,eps,max_iter = 1e4):

#Initial variables
    n = len(A)
    lu = LU_factorization(A)
    y = np.ones(n)
    l_star = 1
    iterations = []
    errors = []
    counter = 0
    #Step 2: initial calculations
    s = np.dot(y,y)
    y_norm = np.sqrt(s)
    x = y/y_norm
    while counter < max_iter:
        #Step 3: solving Ay = x for next approximation
        counter +=1
        iterations.append(counter)
        y = LU_solve(lu,x)
        #Step 4: iterative calculations
        s = np.dot(y,y)
        t = np.dot(y,x)
        y_norm = np.sqrt(s)
        x = y/y_norm
        l = t/s
        error = abs(l - l_star)
        errors.append(error)
        #Step 5: convergence test
        if np.linalg.norm(A@x - t/s*x)/np.linalg.norm(x) < eps:
            break
    return [l,x,iterations,errors]

#Test 1: accuracy test

def accuracy(n):
    #Totally wotking, showing pretty much what is expected:)
    epses = [10**(-i) for i in range(3,10)]
    print(epses)
    errors = []
    A = matr_sep(n,1e-2)
    for eps in epses:
        results = INVIT_demo(A,eps,max_iter=1e5)
        errors.append(abs(1 - results[0]))
    plt.plot(epses,errors,'o',label = 'Experimental data')
    plt.plot(epses,epses,label = 'Expected bound', linestyle = 'dashed')
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('$\\varepsilon$')
    plt.grid()
    plt.loglog()
    plt.show()

#Test 2: distinguishability of the eigenvalues of the matrix.

def convegrence(n,sep1,sep2,eps = 1e-10):
    #Cannot explain the weird behavior of the badly distinguishable spetres...
    seps = [sep1,sep2]
    iters = []
    errors = []
    for sep in seps:
        a = matr_sep(n,sep)
        results = INVIT_demo(a,eps)
        iters.append(results[2])
        errors.append(results[3])
    plt.plot(iters[0],errors[0],label = f'{sep1}')
    plt.plot(iters[1],errors[1],label = f'{sep2}')
    plt.loglog()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def iterations_dist(n,eps):
    #Working correctly
    seps = np.logspace(np.log10(eps)+1,-1,num= 15)
    iterations = []
    for sep in seps:
        a = matr_sep(n,sep)
        results = INVIT_demo(a,eps,max_iter=1e5)
        iterations.append(results[2][-1])
    plt.plot(seps,iterations,'-o',label = f'{eps},Getting ExperiMENTAL about it')
    plt.loglog()
    plt.ylabel('Iterations')
    plt.xlabel('Separation')
    plt.grid()
    plt.legend()
    plt.show()

convegrence(10,1e-2,1e-3)



git config --global user.name "Edmund Charles"

git config --global user.email "ilyadobrygreen@gmail.com"