import matplotlib.pyplot as plt
import numpy as np
from aux_functions import LU_solve, LU_factorization, matr_sep, perturb_matrix_random, matr_spectrum

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
    #UPD: Actually the behavior isn't that weird - the convergence is quadratic (follows from the Raleyigh formula)
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

#Tests for 10 points :D

def perturbations_in_A(n,eps,param: str):
    #Here I shall implement two or more graphs for different 'perekos' coefficients and different distinguishable spectres.
    # That's a lot to do.. but I'll manage it.. 
    #!!The comments below don't really have much weight, since I haven't inspected the 'perekos' effect and the 'bad' spectres. 
    perturbations = np.logspace(-9,-3,num=50)
    spectrum = np.linspace(1,n,n)
    data = []
    errors = []
    a = matr_spectrum(spectrum)
    x_star = INVIT_demo(a,eps)[1]
    for p in perturbations:
        a_p = perturb_matrix_random(a,p)
        results = INVIT_demo(a_p,eps)
        data.append(results)
    if param == 'vector':
        #by the way, not at all sure whether my x_star is actually that accurate, better use some other way of generating
        #the matrix (with a specified eigenvector coresponding to the min lambda)
        #It's pretty stable as well. Nothing catastrophical going on here as well.
        for i in data:
            x = i[1]
            err = x_star - x
            rel = np.linalg.norm(err)/np.linalg.norm(x_star)
            errors.append(rel)
    elif param == 'value':
        #Judjing on what I see from the plot the APEVV is numerically stable..well kind of.. small perturbations = small error
        #It definetly gets worse with bigger condition numbers.. but as not catastrophically as with LS
        for i in data:
            l = i[0]
            err = abs(l - 1)
            errors.append(err)
    else:
        raise ValueError('Enter only supported plot types: either vector or value')
    plt.plot(perturbations,errors,'-o', label = 'Experimental data')
    plt.loglog()
    plt.ylabel('Error')
    plt.xlabel('Perturbations')
    plt.grid()
    plt.legend()
    plt.show()


perturbations_in_A(10,1e-6,'vector')