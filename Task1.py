from matplotlib import pyplot as plt
import numpy as np

func = lambda x: x**4-x**3-7*x**2-8*x-6
phi = lambda x: (x**3+7*x**2+8*x+6)**0.25
def bisection_method(a,b,epsilon,f):
    i = 0 
    errors = []
    while b-a >= 2*epsilon:
        i += 1
        errors.append(b-a)
        c = (a+b)/2
        if f(a)*f(c)< 0:
            b = c
        elif f(c) == 0:
            break
        else :
            a = c
        
    x = (a+b)/2
    return x, i, errors

def simple_interation_method(a,b,epsilon,fi):
    i = 0
    x0 = (a+b)/2
    errors = []
    while True:
        i += 1
        x_next = fi(x0)
        errors.append(abs(x_next-x0))
        if abs(x_next-x0)< epsilon:
            break
        x0 = x_next
    return x0, i, errors
a,b = 3,4
root1, i1, errors1 = bisection_method(a,b,0.00001,func)
root2, i2, errors2 = simple_interation_method(a,b,0.00001,phi)
x = np.linspace(a,b,100)
y = func(x)
it1 = [k for k in range(i1)]
it2 = [k for k in range(i2)]
fig, ax = plt.subplots()
ax.plot(it1,errors1,color= 'green',label='МПД')
ax.plot(it2,errors2,color='red',label='МПИ')
ax.set_title('Зависимость погрешности от количества итераций')
ax.set_xlabel('Количество итераций')
ax.set_ylabel('Погрешность')
ax.set_xlim(0)
ax.set_ylim(0)
ax.grid(True)
ax.legend()
plt.show()