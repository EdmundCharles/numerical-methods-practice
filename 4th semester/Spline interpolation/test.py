import numpy as np
from Cubic_spline import cubic_spline
def thomas_for_spline(x, y,sda = 1,sdb = 1):
    n = len(x)
    
    h = np.array([x[i] - x[i-1] for i in range(1, n)])
    
    b = np.zeros(n)
    c = np.ones(n) * 2
    d = np.zeros(n)
    r = np.zeros(n)
    c[0] = 1
    c[n-1] = 1
    r[0] = sda
    r[n-1] = sdb
    

    for i in range(1, n-1):
        b[i] = h[i-1] / (h[i-1] + h[i])
        d[i] = h[i] / (h[i-1] + h[i])
        r[i] = 6 / (h[i-1] + h[i]) * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    return [b,c,d,r]
def thomas_for_spline_2(x, y,sda = 1,sdb = 1):
    n = len(x)
    
    h = np.array([x[i] - x[i-1] for i in range(1, n)])
    
    b = np.zeros(n)
    c = np.ones(n) * 2
    d = np.zeros(n)
    r = np.zeros(n)
    c[0] = 1
    c[n-1] = 1
    r[0] = sda
    r[n-1] = sdb
    

    for i in range(1, n-1):
        b[i] = h[i-1] / (h[i-1] + h[i])
        d[i] = h[i] / (h[i-1] + h[i])
        r[i] = 6 / (h[i-1] + h[i]) * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
    deltas = np.zeros(n)
    lambdas = np.zeros(n)
    
    # Forward
    deltas[0] = -d[0] / c[0]
    lambdas[0] = r[0] / c[0]
    
    for i in range(1, n):
        den = b[i] * deltas[i-1] + c[i]
        deltas[i] = -d[i] / den
        lambdas[i] = (r[i] - b[i] * lambdas[i-1]) / den
    # return [deltas , lambdas]
    # Back substitution
    ms = np.zeros(n)
    ms[n-1] = lambdas[n-1] 

    for i in range(n-2, -1, -1):
        ms[i] = deltas[i] * ms[i+1] + lambdas[i]
    return ms
def cs(x,y,ms):
    n = len(x)
    h = np.array([x[i] - x[i-1] for i in range(1, n)])
    c_waves = np.ones(n)
    c_s = np.ones(n)
    for i in range(1,n):
        c_waves[i] = y[i-1] - ms[i-1]*h[i-1]**2/6
        c_s[i] = (y[i]-y[i-1])/h[i-1] - h[i-1]/6*(ms[i]-ms[i-1])
    return (c_waves,c_s)
f = lambda x: x - 4*np.cos(x) +5
nodes = np.array([0,1,2,3,4])
d2f = lambda x: 4*np.cos(x)
matr = thomas_for_spline(nodes, f(nodes),d2f(0), d2f(4))
names = ['b','c','d','r']

# for el,name in zip(matr,names):
    # print(f"{name}: {el}")
# print(f(nodes))
# print(d2f(0),d2f(4))
# print(thomas_for_spline_2(nodes, f(nodes),d2f(0), d2f(4)))
# print(cs(nodes,f(nodes),thomas_for_spline_2(nodes, f(nodes),d2f(0), d2f(4))))

def evaluate_cubic_spline(x_array):
    """
    Вычисляет значения конкретного кубического сплайна на отрезке [0, 4].
    
    Параметры:
    x_array (numpy.ndarray или список): Массив точек x, в которых нужно вычислить сплайн.
    
    Возвращает:
    numpy.ndarray: Массив значений сплайна g(x) в заданных точках.
    """
    x = np.asarray(x_array, dtype=float)
    y = np.zeros_like(x)
    
    # Маски для каждого интервала
    mask1 = (x >= 0) & (x < 1)
    mask2 = (x >= 1) & (x < 2)
    mask3 = (x >= 2) & (x < 3)
    # Для последнего интервала включаем правую границу
    mask4 = (x >= 3) & (x <= 4)
    
    # Вычисление g1(x) на [0, 1)
    x1 = x[mask1]
    y[mask1] = (4.0000 / 6) * (1 - x1)**3 + (2.4345 / 6) * x1**3 + 3.0997 * x1 + 0.3333
    
    # Вычисление g2(x) на [1, 2)
    x2 = x[mask2]
    y[mask2] = (2.4345 / 6) * (2 - x2)**3 - (1.8158 / 6) * (x2 - 1)**3 + 5.5342 * (x2 - 1) + 3.4330
    
    # Вычисление g3(x) на [2, 3)
    x3 = x[mask3]
    y[mask3] = -(1.8158 / 6) * (3 - x3)**3 - (4.3535 / 6) * (x3 - 2)**3 + 3.7183 * (x3 - 2) + 8.9672
    
    # Вычисление g4(x) на [3, 4]
    x4 = x[mask4]
    y[mask4] = -(4.3535 / 6) * (4 - x4)**3 + (2.6145 / 6) * (x4 - 3)**3 - 0.6352 * (x4 - 3) + 12.6856
    
    return y

print(np.max(cubic_spline(nodes, f(nodes),d2f(0), d2f(4))[0] - evaluate_cubic_spline(cubic_spline(nodes, f(nodes),d2f(0), d2f(4))[1])))