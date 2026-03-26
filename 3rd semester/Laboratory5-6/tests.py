import matplotlib.pyplot as plt
import numpy as np
from Leverier import get_spectrum
from aux_functions import LU_solve, LU_factorization, matr_sep, perturb_matrix_random, matr_spectrum, get_matr_with_mu, get_mu

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
    epses = np.logspace(-10,-2)
    print(epses)
    errors = []
    A = matr_sep(n,1e-2)
    for eps in epses:
        results = INVIT_demo(A,eps,max_iter=1e5)
        errors.append(abs(1 - results[0]))
    plt.plot(epses,errors,'-o',label = 'Experimental data',markersize = '3')
    plt.plot(epses,epses,label = 'Expected bound', linestyle = 'dashed')
    plt.legend(fontsize = 18)
    plt.ylabel('Error',fontsize = 20)
    plt.xlabel('$\\varepsilon$',fontsize = 20)
    plt.grid()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
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
    plt.plot(iters[0],errors[0],label = f'sep = {sep1}')
    plt.plot(iters[1],errors[1],label = f'sep = {sep2}')
    plt.loglog()
    plt.xlabel('Iterations',fontsize = 20)
    plt.ylabel('Error',fontsize = 20)
    plt.grid()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()

def iterations_dist(n,eps):
    #Working correctly
    seps = np.logspace(np.log10(eps)+1,-1,num= 15)
    iterations = []
    for sep in seps:
        a = matr_sep(n,sep)
        results = INVIT_demo(a,eps,max_iter=1e5)
        iterations.append(results[2][-1])
    plt.plot(seps,iterations,'-o',label = f'$\\varepsilon = ${eps}')
    plt.loglog()
    plt.ylabel('Iterations',fontsize = 20)
    plt.xlabel('Separation',fontsize = 20)
    plt.grid()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()

#Tests for 10 points :D

def perturbations_in_A(n,eps,params: tuple):
    #Here I shall implement two or more graphs for different 'perekos' coefficients and different distinguishable spectres.
    # That's a lot to do.. but I'll manage it.. 
    #!!The comments below don't really have much weight, since I haven't inspected the 'perekos' effect and the 'bad' spectres. 
    #Perecos - lambda , otdel - vec
    perturbations = np.logspace(-9,-3,num=50)
    spectrum = np.linspace(1,n,n)
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    errors1 = []
    errors2 = []
    errors3 = []
    errors4 = []
    if params[0] == 'vector':
        sep1 = params[1]
        sep2 = params[2]
        a1, mu1 = matr_sep(n,sep1), 1
        spectrum = np.linalg.eig(a1).eigenvalues
        a2, mu2 = get_matr_with_mu(spectrum,1000)
        a3, mu3 = matr_sep(n,sep2), 1
        spectrum = np.linalg.eig(a3).eigenvalues
        a4, mu4 = get_matr_with_mu(spectrum,1000)
        x1 = INVIT_demo(a1,eps)[1]
        x2 = INVIT_demo(a2,eps)[1]
        x3 = INVIT_demo(a3,eps)[1]
        x4 = INVIT_demo(a4,eps)[1]
        
        for p in perturbations:
            a_p1 = perturb_matrix_random(a1,p)
            a_p2 = perturb_matrix_random(a2,p)
            a_p3 = perturb_matrix_random(a3,p)
            a_p4 = perturb_matrix_random(a4,p)
            data1.append(INVIT_demo(a_p1,eps))
            data2.append(INVIT_demo(a_p2,eps))
            data3.append(INVIT_demo(a_p3,eps))
            data4.append(INVIT_demo(a_p4,eps))
        # print(data1[0][1])
        print(x1)
        for i in data1:
            x = i[1]
            
            dx = x - x1
            err = np.linalg.norm(dx)/np.linalg.norm(x1)
            errors1.append(err)
        for i in data2:
            x = i[1]
            dx = x - x2
            err = np.linalg.norm(dx)/np.linalg.norm(x2)
            errors2.append(err)
        for i in data3:
            x = i[1]
            dx = x - x3
            err = np.linalg.norm(dx)/np.linalg.norm(x3)
            errors3.append(err)
        for i in data4:
            x = i[1]
            dx = x - x4
            err = np.linalg.norm(dx)/np.linalg.norm(x4)
            errors4.append(err)
        plt.plot(perturbations,errors1,'-o',label = f'$\mu_1 = {round(mu1,0)}$, sep = {sep1}')
        plt.plot(perturbations,errors2,'-o',label = f'$\mu_1 \\approx {round(mu2,0)}$, sep = {sep1}')
        plt.plot(perturbations,errors3,'-*',label = f'$\mu_1 = {round(mu3,0)}$, sep = {sep2}')
        plt.plot(perturbations,errors4,'-*',label = f'$\mu_1 \\approx {round(mu4,0)}$, sep = {sep2}')   
        plt.plot(perturbations,mu2/sep1*perturbations,linestyle = 'dashed', label = 'Theoretical bounds')

    elif params[0] == 'value':
        a1,mu1 = get_matr_with_mu(spectrum,1000)
        a2,mu2 = matr_spectrum(spectrum), 1
        for p in perturbations:
            a_p1 = perturb_matrix_random(a1,p)
            a_p2 = perturb_matrix_random(a2,p)
            data1.append(INVIT_demo(a_p1,eps*1e-3))
            data2.append(INVIT_demo(a_p2,eps*1e-3))
        for i in data1:
            l = i[0]
            err = abs(l - 1)
            errors1.append(err)
        for i in data2:
            l = i[0]
            err = abs(l-1)
            errors2.append(err)
        plt.plot(perturbations,errors1,'-o', label = f'Experimental data: $\mu_1 \\approx$ {round(mu1,0)}')
        plt.plot(perturbations,errors2,'-*', label = f'Experimental data: $\mu_1 =$ {mu2}')
        plt.plot(perturbations,mu1*perturbations,linestyle = 'dashed',label = f'Theoretical bound for $\mu_1 \\approx$ {round(mu1,0)} ')
        # plt.plot(perturbations,mu2*perturbations,linestyle = 'dashed',label = f'Theoretical bound for $\mu_1 \\approx$ {round(mu2,0)} ')
    else:
        raise ValueError('Enter only supported plot types: either vector or value')
    
    plt.loglog()
    plt.ylabel('Error',fontsize = 20)
    plt.xlabel('Perturbations',fontsize = 20)
    plt.grid()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()


# accuracy(10)
# convegrence(10,1e-3,1,eps=1e-14)
# iterations_dist(10,1e-6)
# perturbations_in_A(10,1e-7,('value',1e-2,1))

def accuracy_dim(eps):
    dims = [i for i in range(1,17)]
    errors = []
    lambdass =[0]*dim
    for dim in dims:
        spectrum = np.linspace(1,dim,dim)
        print('Exact\n',spectrum)
        a = matr_spectrum(spectrum)
        for i in range(10):
            lambdas = get_spectrum(a,eps)
            lambdass[i] = lambdas
        for i in lambdass:
            lambdass
        print('Numerical\n',lambdas)
        err = spectrum[0] - lambdas[0]
        error = np.mean(err)
        print(err)
        errors.append(error)
    plt.plot(dims,errors)
    plt.xlabel('dim',fontsize = 20)
    plt.ylabel('Error',fontsize = 20)
    plt.semilogy()
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.ylim(1e-16)
    plt.show()


# accuracy(10)
# convegrence(10,1e-3,1)
# iterations_dist(10,1e-6)
perturbations_in_A(10,1e-10,('vector',1e-4,1))