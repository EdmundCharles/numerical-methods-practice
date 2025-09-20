import matplotlib.pyplot as plt
import numpy as np
import math
trans_fi = lambda x: np.e**(0.5*(1-(np.log10(x))**2))-1
trans = lambda x: 3*(np.log10(x))**2+6*np.log(x+1)-3 
algebraic = lambda x: x**4-3*x**3+3*x**2-2
algebraic_fi = lambda x: (3*x**3-3*x**2+2)**0.25
demo = lambda x: (3-x)**(1/3)
class Solvers:
    def __init__(self):
        self.name: str
        self.root: float
        self.i: int
        self.err: list
    def bisection_method(self,a,b,tr,eps,f,max_iter= 1000):
        self.name = "Bisection method"
        c = (a+b)/2
        i = 0
        err = []
        x = []
        while abs(b-a)>2*eps and i < max_iter:
            i += 1
            c = (a+b)/2
            x.append(c)
            err.append(abs(c-tr))
            if f(a)*f(c)<0:
                b = c
            elif f(c) == 0:
                break
            else:
                a = c
        self.root = float((a+b)/2)
        self.i = i
        self.err = err

    def wegsteins_method_orig(self,x0,tr,eps,f,max_iter = 1000):
        self.name = "Wegstein's method"
        i = 0
        err = []
        x_n = x0
        while i< max_iter:
            i += 1
            if i == 1:
                xnext = f(x_n)
                x_prev = x0
                x_n = xnext
                xn = xnext
                continue
            xnext = f(x_n)
            p = (xnext-xn)/(x_n-x_prev)
            q = p/(p-1)
            x_next = q*x_n + (1-q)*xnext
            err.append(abs(float(tr-x_next)))
            if abs(x_next-x_n)<eps and abs(x_next - f(x_next))<eps:
                break
            x_prev = x_n
            x_n = x_next
            xn = xnext
        self.root = float(x_next)
        self.i = i
        self.err = err
    def wegsteins_method(self,x0,tr,eps,f,max_iter = 1000):
        self.name = "Wegstein's method"
        i = 0
        err = []
        x_n = x0
        while i< max_iter:
            i += 1
            if i == 1:
                xnext = f(x_n)
                x_prev = x0
                x_n = xnext
                xn = xnext
                continue
            xnext = f(x_n)
            x_next = (xnext*x_prev-xn*x_n)/(xnext+x_prev-xn-x_n)
            err.append(abs(float(tr-x_next)))
            if abs(x_next-x_n)<eps :
                break
            x_prev = x_n
            x_n = x_next
            xn = xnext
        self.root = float(x_next)
        self.i = i
        self.err = err
    def wegsteins_method_demo(self,x0,tr,eps,f,max_iter = 1000):
        self.name = "Wegstein's method"
        i = 0
        err = []
        x_n = x0
        while i< max_iter:
            i += 1
            
            if i == 1:
                xnext = f(x_n)
                x_prev = x0
                x_n = xnext
                xn = xnext
                continue
            print('iteration',i)
            xnext = f(x_n)
            print('xn+1=',xnext)
            x_next = (xnext*x_prev-xn*x_n)/(xnext+x_prev-xn-x_n)
            print('x_n+1 = ',x_next)
            err.append(abs(float(tr-x_next)))
            if abs(x_next-x_n)<eps and abs(x_next - f(x_next))<eps:
                break
            x_prev = x_n
            print('x_n-1 = ',x_prev)
            x_n = x_next
            print('x_n = ',x_n)
            xn = xnext
        self.root = float(x_next)
        self.i = i
        self.err = err
class Plotters:
    def convergence(self,method,a,b,tr,eps,f):
        solver = Solvers()
        if method == 'Wegstein':
            solver.wegsteins_method((a+b)/2,tr,eps,f)
            x = [i for i in range(solver.i-1)]
        elif method == 'Bisection':
            solver.bisection_method(a,b,tr,eps,f)
            x = [i for i in range(solver.i)]
        else:
            print('Choose a valid method name')
        y = solver.err
        plt.plot(x,y,'-o',color='red')
        plt.grid(True)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.semilogy()
        plt.show()
    def convergence_2(self,a,b,tr,eps,f,phi):
        solver = Solvers()
        solver.wegsteins_method((a+b)/2,tr,eps,phi)
        x1 = [i for i in range(solver.i-1)]
        y1 = solver.err
        solver.bisection_method(a,b,tr,eps,f)
        x2 = [i for i in range(solver.i)]
        y2 = solver.err
        plt.plot(x1,y1,'-o',color='red',label = 'Wegstein')
        plt.plot(x2,y2,'-*',label = 'Bisection')
        plt.grid(True)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.semilogy()
        plt.legend()
        plt.show()    
    
    
    
    
    
    
    def computational_difficulty(self,method,a,b,f,tr=0):
        solver = Solvers()
        eps_list = [10**i for i in range(-10,0)]
        i_list = [] 
        if method == 'Wegstein': 
            for eps in eps_list:
                solver.wegsteins_method((a+b)/2,tr,eps,f)
                i_list.append(solver.i)
        elif method == 'Bisection':
            for eps in eps_list:
                solver.bisection_method(a,b,tr,eps,f)
                i_list.append(solver.i)
        else:
            print('Choose a valid method name')
        x = eps_list
        y = i_list
        plt.plot(x,y,'-o',color='red')
        plt.grid(True)
        plt.xlabel('$\\varepsilon$')
        plt.ylabel('Iterations')
        plt.semilogx()
        plt.show()
    def computational_difficulty_2(self,a,b,f,phi,tr=0):
        solver = Solvers()
        eps_list = [10**i for i in range(-10,0)]
        i_list1 = [] 
        i_list2 = []
        for eps in eps_list:
                solver.wegsteins_method((a+b)/2,tr,eps,phi)
                i_list1.append(solver.i)
        for eps in eps_list:
                solver.bisection_method(a,b,tr,eps,f)
                i_list2.append(solver.i)
        x = eps_list
        y1 = i_list1
        y2 = i_list2
        plt.plot(x,y1,'-o',label = 'Wegstein')
        plt.plot(x,y2,'-*',label = 'Bisection')
        plt.grid(True)
        plt.xlabel('$\\varepsilon$')
        plt.ylabel('Iterations')
        plt.semilogx()
        plt.legend()
        plt.show()





    def accuracy(self,method,a,b,tr,f):
        solver = Solvers()
        eps_list = [10**i for i in range(-10,-1)]
        err_list = []
        if method == 'Wegstein': 
            for eps in eps_list:
                solver.wegsteins_method((a+b)/2,tr,eps,f)
                if len(solver.err)>=1:
                    err_list.append(solver.err[-1])
        elif method == 'Bisection':
            for eps in eps_list:
                solver.bisection_method(a,b,tr,eps,f)
                print(len(solver.err),solver.i)
                if len(solver.err) >= 1:
                    err_list.append(solver.err[-1])
        else:
            print('Choose a valid method name')
        x = eps_list
        y = err_list
        plt.plot(x,y,'-o',color='red')
        plt.plot(x,x)
        plt.grid(True)
        plt.xlabel('$\\varepsilon$')
        plt.ylabel('Final accuracy')
        plt.loglog()
        plt.show()

    def accuracy_2(self,a,b,tr,f,phi):
        solver = Solvers()
        eps_list = [10**i for i in range(-10,-1)]
        err_list1 = []
        err_list2 = []
        for eps in eps_list:
            solver.wegsteins_method((a+b)/2,tr,eps,phi)
            if len(solver.err)>=1:
                err_list1.append(solver.err[-1])
        y1 = err_list1
        for eps in eps_list:
            solver.bisection_method(a,b,tr,eps,f)
            print(len(solver.err),solver.i)
            if len(solver.err) >= 1:
                err_list2.append(solver.err[-1])
        y2 = err_list2
        x = eps_list
        plt.plot(x,y1,'-o',label = 'Wegstein')
        plt.plot(x,y2,'-*',label = 'Bisection')
        plt.plot(x,x)
        plt.grid(True)
        plt.xlabel('$\\varepsilon$')
        plt.ylabel('Final accuracy')
        plt.loglog()
        plt.legend()
        plt.show()
    
    
    def initial_guess_effect(self,method,a,b,eps,f,tr=0):
        solver = Solvers()
        x_list = np.linspace(a,b,100)
        i_list = []
        if method == 'Wegstein':
            for x in x_list:
                solver.wegsteins_method(x,tr,eps,f)
                i_list.append(solver.i)
        else:
            print('Choose a valid method name')
        y = i_list
        plt.plot(x_list,y,'-o') 
        plt.grid(True)
        plt.xlabel('$x_0$')
        plt.ylabel('Iterations')
        plt.show()
        
solver1 = Solvers()
solver2 = Solvers()
plotter = Plotters()
solver1.bisection_method(1,2,0,1e-6,algebraic)
print(round(solver1.root,7))
solver1.wegsteins_method(1.5,0,1e-6,algebraic_fi)
print(round(solver1.root,7))
solver2.bisection_method(0.4,1,0,1e-6,trans)
print(round(solver2.root,7))
solver2.wegsteins_method(1.5,0,1e-6,trans_fi)
print(round(solver2.root,7))
# ###Algebraic


#Convergence
# plotter.convergence_2(0.4,5,0.61154552014579682218,1e-6,trans,trans_fi)
# plotter.convergence_2(1,2,1.61803398874989,1e-6,algebraic,algebraic_fi)
#Computational_difficulty
plotter.computational_difficulty_2(0.4,1,trans,trans_fi)
# plotter.computational_difficulty_2(1,2,algebraic,algebraic_fi)
# #Accuracy
# plotter.accuracy_2(0.4,1,0.61154552014579682218,trans,trans_fi)
# plotter.accuracy_2(1,2,1.61803398874989,algebraic,algebraic_fi)
# #Initial_guess_effect

plotter.initial_guess_effect('Wegstein',0.4,1,1e-6,trans_fi)
plotter.initial_guess_effect('Wegstein',1,2,1e-6,algebraic_fi)