import numpy as np
import matplotlib.pyplot as plt
from solver import Solver

#Data

f = lambda x,y : (3*x**2*np.e**(-x)-(x+1)*y)/x
a ,b = 1, 5
x0 = a
y0 = 1/np.e
y  = lambda x: x**2*np.e**(-x)
f_solver = Solver(f,a,b,x0,y0,solution = y)

#Tests

def plot_solution(solver,n,eps): #Add different n values
    t_num, y_num = solver.RK2_MP(n)
    t_numa, y_numa = solver.ARK2(eps)
    x_exact,y_exact = solver.exact()
    
    plt.plot(x_exact,y_exact, label = 'Exact Solution',linewidth = 1)
    plt.scatter(t_num,y_num,label = f'Numerical Solution, h = {(solver.b-solver.a)/n}',s = 10, color = 'red')
    plt.scatter(t_numa,y_numa,label = f'Numerical Solution eps = {eps}',s = 10)
    plt.grid()
    plt.legend()
    plt.show()
def error_test(solver,n,eps):
    t_num, y_num = solver.RK2_MP(n)
    errors_n = abs(y_num - solver.solution(t_num))
    t_numa, y_numa = solver.ARK2(eps)
    errors_eps = abs(y_numa - solver.solution(t_numa))
    plt.scatter(t_numa,errors_eps, label = 'Errors',s = 4,color = 'red')
    plt.scatter(t_num, errors_n, label = f'Errors for h = {(solver.b-solver.a)/n}',s = 4, color = 'green')
    plt.axhline(eps, linestyle = '--' ,label = 'Eps')
    plt.grid()
    plt.semilogy()
    plt.legend()
    plt.show()

def convergence_test(solver, n_range):
    local_errors = []
    global_errors = []
    h_range = []
    for n in n_range:
        h = (solver.b-solver.a)/n
        h_range.append(h)
        k1 = solver.f(solver.x0, solver.y0)
        y_local_numeric = solver.y0 + h * solver.f(solver.x0 + h/2, solver.y0 + h/2 * k1)
        y_local_exact = solver.solution(solver.x0 + h)
        local_errors.append(abs(y_local_numeric - y_local_exact))
        
        t_num, y_num = solver.RK2_MP(int(n))
        global_errors.append(abs(y_num[-1] - solver.solution(t_num[-1])))
    

    plt.figure(figsize=(10, 6))
    plt.loglog(h_range, local_errors, 'o', label='Local Error')
    plt.loglog(h_range, global_errors, 's', label='Global Error')
    

    plt.loglog(h_range, [h**3 * local_errors[0]/h_range[0]**3 for h in h_range], '--', color='gray', alpha=0.5, label='$O(h^3)$')
    plt.loglog(h_range, [h**2 * global_errors[0]/h_range[0]**2 for h in h_range], '--', color='gray', alpha=0.7, label='$O(h^2)$')

    plt.xlabel('Step size h')
    plt.ylabel('Error')
    plt.title('Convergence Test: Local vs Global Error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()


# plot_solution(f_solver,n = 100,eps = 1e-5)
# error_test(f_solver,n= 100,eps = 1e-5)
# n_values = np.logspace(1, 5, base = 10)
# convergence_test(f_solver, n_values)


####Second order ODE test

f_sys = lambda t, u: np.array([u[1],u[1]*np.tan(t) - 3*u[0] + np.sin(t)])

y0 = 0
dy0 = 1

a, b = 0, np.pi/2

solution = lambda t: np.array([np.sin(t), np.cos(t)])

system_solver = Solver(f_sys,a,b,0,0,1,solution_sys= solution)

##Modified test

def plot_solution_sys(solver,n):
    t_num, u_num  = solver.RK2_MP_SYS(n)
    t_exact, u_exact = solver.exact_sys()
    print(len(u_exact[0]))

    y_num = u_num[:,0]
    v_num = u_num[:,1]


    y_exact = u_exact[:,0]
    print(y_exact)
    v_exact = u_exact[:,1]

    # plt.plot(t_exact,y_exact, label = 'Exact Solution',linewidth = 1)
    # plt.scatter(t_num,y_num,label = f'Numerical Solution, h = {(solver.b-solver.a)/n:.4f}',s = 10, color = 'red')


    # plt.plot(t_exact,v_exact, label = 'Exact Solution',linewidth = 1)
    # plt.scatter(t_num,v_num,label = f'Numerical Solution, h = {(solver.b-solver.a)/n}',s = 10, color = 'red')

    plt.plot(y_num,v_num)

    plt.grid()
    plt.legend()
    plt.show()

# plot_solution_sys(system_solver,n = 100)

# print(system_solver.exact_sys())

