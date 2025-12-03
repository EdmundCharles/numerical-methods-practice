import matplotlib.pyplot as plt
import numpy as np

def sorter(coefficients):
    n = len(coefficients)
    if (n & (n - 1)) != 0 or n < 1:
        raise ValueError("n must be a power of 2")
    
    if n == 1:
        return coefficients
    
    permutation = [1, 2]
    current_size = 2
    
    while current_size < n:
        next_size = current_size * 2
        new_permutation = []
        for j in permutation:
            new_permutation.append(j)
            new_permutation.append(next_size + 1 - j)
        permutation = new_permutation
        current_size = next_size
    
    return [coefficients[j-1] for j in permutation]
def matr_cond(n, cond):
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
    l = np.linspace(1, cond, n)
    d = np.diag(l)
    q = np.linalg.qr(np.random.rand(n, n)).Q
    a = np.matmul(np.matmul(q, d), np.linalg.matrix_transpose(q))
    return a
def get_alphas(m, lmin, lmax, sort):
    lowr, uppr = lmin, lmax
    alphas = [0]*m
    for k in range(1, m+1):
        t = np.cos(np.pi*(2*k-1)/(2*m))
        l = (lowr+uppr)/2 + (uppr-lowr)/2*t
        alphas[k-1] = 1/l
    print(len(alphas))
    if sort:
        # alphas.sort(reverse=True, key=lambda x: abs(x))
        alphas = sorter(alphas)
    return alphas



def err_iter(n, cond, m=10, eps=1e-7, max_iter=1e4):
    A = matr_cond(n, cond)
    x_star = np.ones(n)
    b = A @ x_star
    alphas = get_alphas(m, 1, cond,False)
    x = np.zeros(n)
    count = 0
    errors = []
    iters = []
    while count < max_iter:
        for k in range(m):
            alpha = alphas[k]
            residual = A @ x - b
            x_next = x - alpha*residual
            x = x_next

            residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
            err = np.linalg.norm(x-x_star)/np.linalg.norm(x_star)
            errors.append(err)
            count += 1
            iters.append(count)
        if residual_norm*cond < eps:
            print(f'Требуемая точность достигнута за {count} серий')
            break
        print(f"Restart {count + 1}: ||r|| = {residual_norm:.2e}")

    plt.plot(iters, errors, color='red')
    plt.ylabel('$\\frac{||\\delta x||}{||x||}$', loc='center',rotation='horizontal',labelpad= 10,size = 18)
    plt.xlabel('Iterations',size = 14)

    plt.loglog()
    plt.tight_layout()
    plt.grid()
    plt.show()


def i_cond(n, eps=1e-4, m=10, max_iter=1e4):
    conds = np.logspace(3, 10, 10)
    iters = []
    for cond in conds:
        a = matr_cond(n, cond)
        alphas = get_alphas(m, 1, cond, True)
        x_star = np.ones(n)
        b = a @ x_star
        x = np.zeros(n)
        count = 0
        while count < max_iter:
            count += 1
            for k in range(m):
                alpha = alphas[k]
                residual = a @ x - b
                x_next = x - alpha*residual
                x = x_next

                residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
                err_approx = residual_norm*cond

            if err_approx < eps:
                print(f'Требуемая точность достигнута за {count} серий')
                break
            print(f"Рестарт {count + 1}: ||r|| = {err_approx:.2e}")
        iters.append(count*m)
    plt.plot(conds, iters, color='red')
    plt.ylabel('Iterations required', loc='center',
               rotation='vertical', labelpad=20,size = 14)
    plt.xlabel('Condition number',size = 14)
    plt.loglog()
    plt.tight_layout()
    plt.grid()
    plt.show()


def iter_m(n, cond1,cond2, eps, max_iter=10000):
    ms = [2**i for i in range(1,6+4)]
    print(ms)
    iters11 = []
    iters12 = []
    iters21 = []
    iters22 = []
    a1 = matr_cond(n, cond1)
    a2 = matr_cond(n, cond2)
    x_star = np.ones(n)
    b1 = a1 @ x_star
    b2 = a2 @ x_star 
    
    #COND1

    for m in ms:
        count = 0
        alphas = get_alphas(m, 1, cond1,sort=False)  
        x = np.zeros(n)
        converged = False
        
        while count < max_iter and not converged:
            count += 1
            

            for k in range(m):
                
                alpha = alphas[k]
                residual = a1 @ x - b1
                x = x - alpha * residual  
                
                residual_norm = np.linalg.norm(residual) / np.linalg.norm(b1)
                

                if residual_norm*cond1 < eps:
                    converged = True
                    break

        iters11.append(count*m)
        # print(f"m = {m}: {count} итераций, сходимость = {converged}")
    for m in ms:
        count = 0
        alphas = get_alphas(m, 1, cond1,sort= True)  
        x = np.zeros(n)
        converged = False
        
        while count < max_iter and not converged:
            count += 1
            

            for k in range(m):
                
                alpha = alphas[k]
                residual = a1 @ x - b1
                x = x - alpha * residual  
                
                residual_norm = np.linalg.norm(residual) / np.linalg.norm(b1)
                

                if residual_norm*cond1 < eps:
                    converged = True
                    break

        iters12.append(count*m)
        # print(f"m = {m}: {count} итераций, сходимость = {converged}")
    
    #COND2
    
    for m in ms:
        count = 0
        alphas = get_alphas(m, 1, cond2,sort=False)  
        x = np.zeros(n)
        converged = False
        
        while count < max_iter and not converged:
            count += 1
            

            for k in range(m):
                
                alpha = alphas[k]
                residual = a2 @ x - b2
                x = x - alpha * residual  
                
                residual_norm = np.linalg.norm(residual) / np.linalg.norm(b2)
                

                if residual_norm*cond2 < eps:
                    converged = True
                    break
        iters21.append(count*m)
    for m in ms:
        count = 0
        alphas = get_alphas(m, 1, cond2,sort = True)  
        x = np.zeros(n)
        converged = False
        
        while count < max_iter and not converged:
            count += 1
            

            for k in range(m):
                
                alpha = alphas[k]
                residual = a2 @ x - b2
                x = x - alpha * residual  
                
                residual_norm = np.linalg.norm(residual) / np.linalg.norm(b2)
                

                if residual_norm*cond2 < eps:
                    converged = True
                    break
        iters22.append(count*m)
        # print(f"m = {m}: {count} итераций, сходимость = {converged}")
    plt.plot(ms, iters11, 'o-', markersize=3,label = f'cond = $10^{{{np.log10(cond1)}}}$, no sorting')
    plt.plot(ms,iters12,'o-',markersize = 3,label = f'cond = $10^{{{np.log10(cond1)}}}$, sorted')
    plt.plot(ms, iters21, 'o-', markersize=3,label = f'cond = $10^{{{np.log10(cond2)}}}$, no sorting')
    plt.plot(ms,iters22,'o-',markersize = 3,label = f'cond = $10^{{{np.log10(cond2)}}}$, sorted')
    plt.axhline(512*10000,linestyle = 'dashed',label = 'MAX iterations =  $512\cdot 10^4$', color = 'red')
    plt.xlabel('$m$',size = 14)
    plt.ylabel('Iterations',size = 14)
    plt.semilogy()
    plt.ylim(1e1,1e7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    

# err_iter(100,100,m=5,eps = 1e-12)
# i_cond(100,eps = 1e-6,m = 256)
# iter_m(100,10,1e5,1e-6)
