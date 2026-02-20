import numpy as np

def thomas_for_spline(x, y,sda = 1,sdb = 1):
    n = len(x)
    
    h = np.array([x[i] - x[i-1] for i in range(1, n)])
    
    b = np.zeros(n)
    c = np.ones(n) * 2
    d = np.zeros(n)
    r = np.zeros(n)
    
    c[0] = sda
    c[n-1] = sdb
    

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

def gs(x,ms,c_waves,c_s):
    n = len(x)
    h = np.array([x[i] - x[i-1] for i in range(1, n)])
    g = []
    ab = []
    def g_i(m1,m2,x1,x2,h,c_wave,c,x):
        return m1*(x2-x)**3/(6*h) + m2*(x-x1)**3/(6*h) + c*(x - x1) + c_wave
    for i in range(1,n):
        interval = np.linspace(x[i-1],x[i],20)
        ab.append(interval)
        # g[i] = np.array([g_i(ms[i-1],ms[i],x[i-1],x[i],h[i-1],c_waves[i],c_s[i],x) for x in interval])
        g.append(g_i(ms[i-1],ms[i],x[i-1],x[i],h[i-1],c_waves[i],c_s[i],interval))

    g = np.hstack(g)
    ab = np.hstack(ab)
    return g, ab
