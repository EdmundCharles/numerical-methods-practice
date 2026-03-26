import numpy as np
from mpmath import mp


class Integrator:
    def __init__(self,f,a,b,max_iter = 30):
        self.f = f
        self.a = a
        self.b = b
        self.max_iter = max_iter

        self.xkc = np.array([-1,(1-np.sqrt(6))/5,(1+np.sqrt(6))/5])
        self.Akc = np.array([2/9,(16+np.sqrt(6))/18,(16-np.sqrt(6))/18])
    @staticmethod
    def transform(a,b,xkc,Akc):
        xk = (a + b) / 2 +(b - a) / 2* xkc
        Ak = (b - a) / 2 * Akc
        return xk, Ak
    @staticmethod
    def runge_rule(SN, S2N,p = 5):
        return abs(SN - S2N)/(2**p-1)
    def _s_radau_3(self,a,b):
        xk, Ak = self.transform(a,b ,self.xkc, self.Akc)
        return np.dot(Ak,self.f(xk))
    def radau_3(self,eps):
        i = 0
        N = 1
        result_prev = self._s_radau_3(self.a,self.b)
        calls = 3
        while i < self.max_iter:
            i += 1
            N *= 2
            intervals = np.linspace(self.a,self.b,N+1)
            result_new = np.sum([self._s_radau_3(intervals[j],intervals[j+1]) for j in range(N)])
            calls += N*3
            if self.runge_rule(result_prev, result_new) < eps:
                return result_new, calls
            result_prev = result_new
        raise RuntimeError(f"{self.radau_3.__name__} did not achieve desired accuracy after {i} iterations. Got: {result_new}")
    def exact_val(self):
        mp.dps = 20
        return float(mp.quad(self.f,[self.a,self.b]))