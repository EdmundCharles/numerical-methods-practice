import numpy as np


def Lagrange_legacy(x_nodes,y_nodes):
    """
    Computes the coefficients of the Lagrange polynomial implementing the following formula:

                            L_n(x) = ∑y_i*Ф_i(x)


    Parameters
    ----------
    x_nodes : array_like
        1-D array containing the x-coordinates of the data points. 
        All values must be distinct to avoid division by zero.
    y_nodes : array_like
        1-D array containing the y-coordinates of the data points.

    Returns
    -------
    coefficients : ndarray
        An array of polynomial coefficients.

    """
    def Phi_i(i,x_nodes):
        n = len(x_nodes)
        numerator = [1.0]
        denominator = 1
        for k in range(n):
            if k != i:
                numerator = np.convolve(numerator,[1,-x_nodes[k]])
                denominator *= x_nodes[i]-x_nodes[k]
            else:
                continue
        coeffs = numerator/denominator
        return coeffs
    
    n = len(x_nodes)
    coefficients = np.zeros(n)
    for i in range(n): 
        coefficients += y_nodes[i]*Phi_i(i,x_nodes)
    return coefficients




def Lagrange(x_nodes: list,y_nodes: list,x):
    n = len(x_nodes)
    def get_phi_x(i,x_nodes,x):
        numerator = 1
        denominator = 1
        for k in range(n):    
            if k != i: 
                numerator *= x - x_nodes[k]
                denominator *= x_nodes[i] - x_nodes[k]
        phix = numerator/denominator
        return phix
    phis = np.array([get_phi_x(i,x_nodes,x) for i in range(n)])

    return np.dot(y_nodes,phis)

x_nodes = np.array([1,2,3,4])
f = lambda x: x - np.cos(x) - 1 
y_nodes = f(x_nodes)
 

def test(f,x):
    new_val = Lagrange(x_nodes,y_nodes,x)
    old = Lagrange_legacy(x_nodes,y_nodes)
    old_val = np.polyval(old,x)
    print(f'Погрегность в точке {x} для старой версии: {f(x+0.5) - old_val}')
    print(f'Погрегность в точке {x} для новой версии: {f(x+0.5) - new_val}')
    print(f'Разница: {np.abs(old_val - new_val)}')

