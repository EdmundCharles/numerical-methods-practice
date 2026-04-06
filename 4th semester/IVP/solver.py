import numpy as np


class Solver:
    def __init__(self, f, a, b, x0, y0,dy0 = None , solution=None, solution_sys = None):
        self.f = f
        self.a = a
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self.dy0 = dy0
        self.solution = solution
        self.solution_sys = solution_sys

    @staticmethod
    def runge_rule(yh, y2h, p=2):
        return abs(yh - y2h)/(2**p - 1)

    @staticmethod
    def vec_runge_rule(uh,u2h, p = 2):
        return np.linalg.norm(uh - u2h)/(2**p - 1)
    
    def exact(self):
        x = np.linspace(self.a, self.b, 1000)
        y = self.solution(x)
        return x, y
    
    def exact_sys(self):
        x = np.linspace(self.a,self.b,1000)
        u = self.solution_sys(x)
        return x, u.T

    def RK2_MP(self, n):  # h --> n
        h = (self.b-self.a)/n
        T = np.linspace(self.a, self.b, n+1)
        Y = [self.y0]
        y_prev = self.y0
        for t in T[:-1]:
            y_new = y_prev + h*self.f(t+h/2, y_prev + h/2*self.f(t, y_prev))
            Y.append(y_new)
            y_prev = y_new
        return T, np.array(Y)
    
    def RK2_MP_SYS(self, n):
        if self.dy0 == None:
            raise AttributeError("In order to use this method specify valid initital values for y and y'")
        
        h = (self.b-self.a)/n
        T = np.linspace(self.a, self.b, n+1)
        U = [np.array([self.y0,self.dy0])]
        u_prev = U[0]
        for t in T[:-1]:
            u_new = u_prev + h*self.f(t + h/2, u_prev + h/2*self.f(t, u_prev))
            U.append(u_new)
            u_prev = u_new
        return T, np.array(U,dtype= 'object')
    def ARK2(self, eps):
        # experimetntal
        eps *= 1e-3
        h = 1
        Y = [self.y0]
        T = [self.x0]
        y_curr = self.y0
        t_curr = self.a
        while t_curr < self.b:
            if t_curr + h > self.b:
                h = self.b - t_curr
            # 1h
            k1 = self.f(t_curr, y_curr)
            yh = y_curr + h*self.f(t_curr + h/2, y_curr + h/2 * k1)
            # 0.5h
            h_half = h/2

            k1_half = self.f(t_curr, y_curr)
            yh_half_mid = y_curr + h_half * \
                self.f(t_curr + h_half/2, y_curr + h_half/2*k1_half)

            t_half_mid = t_curr + h_half
            k2_half = self.f(t_half_mid, yh_half_mid)
            y_2half = yh_half_mid + h_half * \
                self.f(t_half_mid + h_half/2, yh_half_mid + h_half/2*k2_half)
            # Runge
            R = self.runge_rule(yh, y_2half)

            if R <= eps:
                y_curr = y_2half
                t_curr += h
                Y.append(y_curr)
                T.append(t_curr)
                if R < eps/10:
                    h *= 2
            else:
                h /= 2

            if h < 1e-12:
                break
        return np.array(T), np.array(Y)
