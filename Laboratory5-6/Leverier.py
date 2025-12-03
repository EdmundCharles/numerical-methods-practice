import numpy as np
from aux_functions import tr

def Leverier(A):
    n = len(A)
    p = np.zeros(n)
    p[0] = 1
    s = np.zeros(n)
    for k in range(1,n+1):
        Ak = np.linalg.matrix_power(A,k)
        s[k-1] = tr(Ak)
        s_ = s[k-1]
        for i in range(1,k):
            s_ -= p[i-1]*s[k-i-1]
        p[k-1] = s_/k
    return p


#The main question here is whether I should implement a sort of real root isolation algorithm (which would be quite cool). 
# In this case it would be a full APEVV

