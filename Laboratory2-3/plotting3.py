import numpy as np
import matplotlib.pyplot as plt
from plotting import error,error2,matr_cond,perturb_matrix_random
from rotations import rotations_py
from GEM import GEM_Py
import timeit
def get_exponent(sizes, times):
        log_n = np.log(sizes)
        log_t = np.log(times)
        return np.polyfit(log_n, log_t, 1)[0]
def error_rot(A,A_dist=None):
    return error(A,rotations_py,A_dist)
def error_gem(A,A_dist=None):
    return error(A,GEM_Py,A_dist)
def nevyazka_rot(A):
    return error2(A,method=rotations_py)
def nevyazka_gem(A):
    return error2(A,method=GEM_Py)
def plot_err_cond_both(n,num):
    eps = 2.2e-16
    cond = np.logspace(1,9,300)
    err_gem = []
    nev_gem = []
    err_rot = []
    nev_rot = []
    max_errors = []
    for i in cond:
        print(i)
        a = matr_cond(n,i)
        err_gem.append(error_gem(a))
        nev_gem.append(nevyazka_gem(a))
        err_rot.append(error_rot(a))
        nev_rot.append(nevyazka_rot(a))
        max_errors.append(i*(2*eps)/(1-eps*i))
    plt.scatter(cond,err_gem,s=1.5,label = 'Error GEM')
    plt.scatter(cond,nev_gem,s = 1.5,label = 'Residual GEM')
    plt.scatter(cond,err_rot,s=1.5,label = 'Error rotations')
    plt.scatter(cond,nev_rot,s = 1.5,label = 'Residual rotations')
    plt.plot(cond,max_errors,linestyle= 'dashed',color = 'red',label = 'Theoretical bound')
    plt.xlabel('$cond(A)$')
    plt.ylabel('$\\frac{||\\delta x||}{||x||}$',loc='center',rotation= 'horizontal',size='14',labelpad=20)
    plt.loglog()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'c:\Users\ilyad\Downloads\Test1'+f'{num}.eps')
    plt.show()
def plot_err_deltaA(n,cond,num):

    epsilons = np.logspace(-2,0,10).tolist()
    a = matr_cond(n,cond)
    errors_gem = []
    errors_rot = []
    max_errors = []
    for eps in epsilons:
        a_eps = perturb_matrix_random(a,eps)
        errors_rot.append(error_rot(a,a_eps))
        errors_gem.append(error_gem(a,a_eps))
    i = 0
    for eps in epsilons:
        if 1 - cond*eps< 1e-12:
            break
        i += 1
        max_errors.append(cond*eps/(1-cond*eps))

    # for eps in epsilons:
    #     a_eps = perturb_matrix_random(a,eps)
    #     errors_gem.append(error_gem(a,a_eps))
    print(errors_gem)
    print(errors_rot)
    # plt.figure(figsize=(6,6))
    plt.scatter(epsilons,errors_gem,s=7,label = 'Experimental data GEM')
    plt.scatter(epsilons,errors_rot,s=1.5,label = 'Experimental data rotations method')
    plt.plot(epsilons[:i],max_errors,linestyle = 'dashed',label='Theoretical bound',color = 'red')
    plt.xlabel('$\\frac{||\\delta A||}{||A||}$',size = 16)
    plt.ylabel('$\\frac{||\\delta x||}{||x||}$',loc='center',rotation= 'horizontal',labelpad=20,size = 16)
    plt.loglog()
    plt.legend()
    plt.grid(True,alpha = 0.3)
    plt.tight_layout()
    plt.savefig(r'c:\Users\ilyad\Downloads\Test2'+f'{num}.eps')
    plt.show()
    

def plot_comp_n(cond):
    times_gem = []
    times_rot = []
    alphas_gem = []
    alphas_rot = []
    dims = []
    for n in range(10,1011,100):
        dims.append(n)
        a = matr_cond(n,cond)
        b = [i for i in range(n)]
        time_gem = timeit.timeit(lambda: GEM_Py(a,b),number=1)
        time_rot = timeit.timeit(lambda: rotations_py(a,b),number=1)
        alpha_gem = time_gem/n**3
        alpha_rot = time_rot/n**3
        times_gem.append(time_gem)
        times_rot.append(time_rot)
        alphas_gem.append(alpha_gem)
        alphas_rot.append(alpha_rot)
        print(n,time_rot,time_gem)
    exp_rot = get_exponent(dims, times_rot)
    exp_gem = get_exponent(dims, times_gem)
    beta_gem = np.mean(alphas_gem)
    beta_rot = np.mean(alphas_rot)
    theor_gem = [i**3*beta_gem for i in dims]
    theor_rot = [i**3*beta_rot for i in dims]
    plt.figure(figsize=(10,6))
    plt.plot(dims,times_gem,label = 'Experimental data GEM')
    plt.plot(dims,times_rot,label = 'Experimental data rotations')
    plt.plot(dims,theor_gem,label = f'$t = {beta_gem*1e8:.2f}\\cdot 10^{{-8}} \\cdot n^{{{exp_gem:.2f}}}$',linestyle = 'dashed')
    plt.plot(dims,theor_rot,label = f'$t = {beta_rot*1e8:.2f}\\cdot 10^{{-8}} \\cdot n^{{{exp_rot:.2f}}}$',linestyle = 'dashed')
    plt.xlabel('$n = dim(A)$')
    plt.ylabel('Execution time (t), c')
    plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    plt.legend()
    plt.loglog()
    plt.show()
# plot_err_cond_both(10,1)
# plot_err_cond_both(100,2)
# plot_err_deltaA(10,1e2,1)
# plot_err_deltaA(100,1e7,2)