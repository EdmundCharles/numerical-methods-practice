import numpy as np

class Methods:
    def decorator(func):
        def wrapper():
            print('Executing ')
            func
        return wrapper
    
    
    def Newton(self,a,b,x0,eps,f,df):
        i = 0
        x_n = x0
        ab = np.linspace(a,b,1000)
        df_on_ab = [df(i) for i in ab]
        m = min(df_on_ab)
        M = max(df_on_ab) 
        while True:
            i += 1 
            x_next = x_n - f(x_n)/df(x_n)
            if M/(2*m)*abs(x_next-x_n)**2< eps:
                break
            x_n = x_next
        return i, x_next
    def secant(self,a,b,x0,eps,f,df):
        i = 0
        x_n = x0
        ab = np.linspace(a,b,1000)
        df_on_ab = [df(i) for i in ab]
        m = min(df_on_ab)
        M = max(df_on_ab) 
        while True:
            i+=1
            x_next = x_n - f(x_n)*(x_n)
    

algebraic = lambda x: x**4-3*x**3+3*x**2-2
df1 = lambda x: 4*x**3-9*x**2+6*x
solver = Methods()
print(solver.Newton(1,2,2,1e-6,algebraic,df1))