import numpy as np
import copy
def rotations_np(A,b):
    n = len(A)
    Ab = np.column_stack((A, b))
    for i in range(0,n):
        for j in range(i+1,n):
            aii = Ab[i][i]
            aji = Ab[j][i]
            sqrt = np.sqrt(aii**2 + aji**2)
            c = aii/sqrt
            s = aji/sqrt
            Ab[i],Ab[j] = c*Ab[i] + s*Ab[j],-s*Ab[i] + c*Ab[j]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (Ab[i][n]- Ab[i][i+1:n] @ x[i+1:n])/Ab[i][i]
    return x

def rotations_py(A,b):
    n = len(A)
    Ab = copy.deepcopy(A)
    bc = copy.deepcopy(b)
    for i in range(n):
        Ab[i].append(bc[i])
    for i in range(0,n):
        for j in range(i+1,n):
            aii = Ab[i][i]
            aji = Ab[j][i]
            sqrt = np.sqrt(aii**2 + aji**2)

            if abs(aji) < 1e-12:
                continue  # Элемент уже нулевой, вращение не нужн
            
            
            if abs(sqrt) < 1e-12:
                continue  # Избегаем деления на ноль
            c = aii/sqrt
            s = aji/sqrt
            for k in range(i,n+1):
                Ab[i][k],Ab[j][k] = c*Ab[i][k] + s*Ab[j][k],-s*Ab[i][k] + c*Ab[j][k]
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = Ab[i][n]
        for j in range(i+1, n):
            x[i] -= Ab[i][j] * x[j]
        x[i] /= Ab[i][i]
    
    return x 