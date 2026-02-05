import numpy as np
from aux_functions import matr_sep, LU_solve, LU_factorization



def INVIT_demonstrator(A,eps,max_iter = 100000):
    #Initial variables
    n = len(A)
    print('Исходная матрица А:\n',A)
    lu = LU_factorization(A)
    print('LU-разложение матрицы A:\n',lu)
    y = np.ones(n)
    print('Начальное приближение собственного вектора:\n y^(0)=',y)
    l = 0
    print('Начальное приближение собственного числа:\n \lambda^(0)=',l)
    counter = 0
    
    #Step 2: initial calculations
    print('Нулевая итерация')
    s = np.dot(y,y)
    print('s^(0) = (y^(0),y^(0)) = ', s)
    y_norm = np.sqrt(s)
    print('||y^(0)|| = sqrt(s^(0))')
    x = y/y_norm
    print('x^(0) = y^(0)/||y^(0)||')
    while counter < max_iter:
        #Step 3: solving Ay = x for next approximation
        counter +=1 
        print('Итерация', counter)
        y = LU_solve(lu,x)
        print(f'Решаем систему Ay^({counter} = x^({counter-1}))')
        print(f'Получено решение y^({counter})= {y}')
        #Step 4: iterative calculations
        s = np.dot(y,y)
        print(f's^({counter} = (y^({counter}),y^({counter}))')
        t = np.dot(y,x)
        print(f't^({counter} = (y^({counter},x^{counter-1})))')
        y_norm = np.sqrt(s)
        print(f'||y^({counter})|| = {y_norm}')
        x = y/y_norm
        print(f'x^({counter}) = y^{counter}/||y^({counter})||')
        l = t/s
        print(f'\lambda^({counter}) = t^({counter})/s^({counter})')
        #Step 5: convergence test
        err = np.linalg.norm(A@x - l*x)/np.linalg.norm(x)
        print('Проверка сходимости\n')
        if err < eps:
            print(f'||A*x^({counter}) - \lambda^({counter})*x||/||x^({counter}) = {err} < {eps}')
            print(f'Точность {eps} достигнута за {counter} итераций')
            print(f'Полученное решение:\n \lambda_1 = {l}, x_1 = {x}')
            break
        print(f'||A*x^({counter}) - \lambda^({counter})*x||/||x^({counter}) = {err} > {eps}')
        print(f'Точность {eps} не достигнута, продолжаем итерации')
    return l ,counter

# A = matr_sep(3,10)
A = np.array([[1,2,],[2,1]])

INVIT_demonstrator(A,1e-3)
