import numpy as np
from typing import Callable, Union
from scipy.integrate import quad
from mpmath import mp
import sympy as sp
import math
#Implementation of trapezoidal integration formula

class Integrator_NC:
    def __init__(self,f: Callable,a: Union[float,int],b: Union[float,int], max_iter: int = 100):
        self._validate_input(f,a,b,max_iter)
        self.f = f
        self.a = a
        self.b = b
        self.max_iter = max_iter
        self.current_sum = None
    def _validate_input(self, f, a, b, max_iter):
        """Internal method to validate input parameters."""
        
        if not callable(f):
            raise TypeError(f"Argument 'f' must be callable. Got: {type(f).__name__}")
        
        if not all(isinstance(x, (int, float, np.number)) for x in [a, b]):
            raise TypeError("Integration boundaries 'a' and 'b' must be numbers (int, float, or np.number)")
            
        # if not isinstance(eps, (int, float)) or eps <= 0:
        #     raise ValueError(f"Tolerance 'eps' must be a positive number. Got: {eps}")
            
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(f"Maximum iterations 'max_iter' must be a positive integer. Got: {max_iter}")

    def _S2(self, N: int):
        x = np.linspace(self.a,self.b,N)
        h = (self.b-self.a)/(N-1)
        y = self.f(x)
        calls = len(x)
        return h/2*(y[0] + 2*y[1:-1].sum() + y[N-1]), h, calls
    def _S2_efficent(self,s_prev,h_prev,intervals_prev):
        h = h_prev/2
        x = np.linspace(self.a+h,self.b-h,intervals_prev)
        y = self.f(x)
        calls = len(x)
        # self._S2_efficent.calls +=1
        return 0.5*s_prev + h*y.sum() , h, intervals_prev*2, calls
    @staticmethod
    def runge_rule(SN, S2N,p = 2):
        return abs(SN - S2N)/(2**p-1)
    
    def trapz_naive(self,eps):
        i = 0
        n = 2
        S_prev, _, total_calls = self._S2(n)
        while i < self.max_iter:
            i += 1
            n *= 2
            S_new, _, calls = self._S2(n)
            total_calls += calls
            if self.runge_rule(S_prev, S_new) < eps:
                return S_new, total_calls
            S_prev = S_new
        raise RuntimeError(f"The {self.trapz_naive.__name__} did not converge after {self.max_iter} iterations, N = {n}")
    def trapz_efficent(self,eps):
        i = 0
        n = 2
        intervals_prev = 1
        S_prev, h_prev, total_calls= self._S2(n)
        while i < self.max_iter:
            i += 1
            S_new, h_prev, intervals_prev,calls = self._S2_efficent(S_prev,h_prev,intervals_prev)
            total_calls += calls
            if self.runge_rule(S_prev, S_new) < eps :
                return S_new, total_calls
            S_prev = S_new
        raise RuntimeError(f"The {self.trapz_naive.__name__} did not converge after {self.max_iter} iterations, N = {n}, got {S_new}")
    def _trapz(self,eps,int_type = 'naive'):
        if int_type == 'naive':
            return self._trapz_naive(eps)
        elif int_type == 'efficent':
            return self._trapz_efficent(eps)
        else:
            raise NameError("Either 'naive' or 'efficient' integrator types are supported for _trapz")
    
    def exact_val(self):
        return quad(self.f,self.a,self.b,epsabs=1e-16,epsrel=1e-16)
    def mp_exact_val(self):
        mp.dps = 35
        return float(mp.quad(self.f,[self.a,self.b]))
    def _s_n(self,n, alp, bet):
        H = np.array(self.cotes_coefs(n))
        x = np.linspace(alp, bet, n)
        y = self.f(x)
        return (bet - alp) * np.dot(H, y)
    def S_n(self,n):
        H = np.array(self.cotes_coefs(n))
        x = np.linspace(self.a, self.b, n)
        y = self.f(x)
        return (self.b - self.a) * np.dot(H, y)

    @staticmethod
    def cotes_coefs(num):

        n, k, t, j = sp.symbols('n k t j')

        omega_bar = sp.prod([t-i for i in range(num)])

        integrand = (omega_bar / (t - k + 1)).expand()

        coefficient = ((-1)**(n - k)) / (sp.factorial(k - 1) * sp.factorial(n - k))

        integral_expr = coefficient * sp.Integral(integrand, (t, 0, n - 1))
        H = [integral_expr.subs({k : ind, n: num}).doit() for ind in range(1,num+1)]
        return H
            

f = lambda x: np.log(x**3-np.sin(x))**(np.cos(x)-np.e**x)   
