import numpy as np
import copy
def GEM_Py(A, b):
    n = len(A)
    Ab = copy.deepcopy(A)
    bc = copy.deepcopy(b)
    for i in range(n):
        Ab[i].append(bc[i])    
    for i in range(n-1):
        max_val = abs(Ab[i][i])
        max_ind = i
        for j in range(i+1,n):
            if abs(Ab[j][i]) > max_val:
                max_val = abs(Ab[j][i])
                max_ind = j
        if max_ind != i:
            Ab[i], Ab[max_ind] = Ab[max_ind], Ab[i]
        for j in range(i+1,n):
            m = Ab[j][i]/Ab[i][i]
            for k in range(i,n+1):
                Ab[j][k] -= m*Ab[i][k]
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = Ab[i][n]
        for j in range(i+1, n):
            x[i] -= Ab[i][j] * x[j]
        x[i] /= Ab[i][i]
    
    return x 
            
def GEM_NP(A, b):
    n = len(A)
    Ab = np.column_stack((A, b))

    # Forward elimination
    for k in range(n-1): 
        # Partial pivoting
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]
        
        # Vectorized elimination
        for i in range(k+1, n):
            if Ab[k, k] == 0:
                continue
            multiplier = Ab[i, k] / Ab[k, k]
            Ab[i, k:] = Ab[i, k:] - multiplier * Ab[k, k:]
    
    # Backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


