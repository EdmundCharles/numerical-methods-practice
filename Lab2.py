import numpy as np
import matplotlib.pyplot as plt
def print_matrix(matrix):
    for row in matrix:
        print(row)
def gemwec(A, bt):
    """
    This function implements the Modified Gaussian elimination method for solving systems of linear equations Ax=b

    Parameters
    ----------
    A : array_like
        (n, n) array representing the coefficient matrix.
    bt : array_like
        (1,n) array representing the transposed free term vector.

    Returns
    -------
    x : array
        The solution vector for the given system of linear equations.
    """
    #Preparation
    n = len(A)
    augmented_matrix = [A[i]+[bt[i]] for i in range(n)]
    Ab = augmented_matrix
    print_matrix(Ab)
    #Forward elimination
    for k in range(n):
        #Finding the largest A[i][k] and swapping the rows
        max_val = abs(Ab[k][k])
        max_row = k
        for i in range(k+1,len(Ab)):
            if abs(Ab[i][k]) > max_val:
                max_val = abs(Ab[i][k])
                max_row = i
        if max_row != k:
            Ab[k], Ab[max_row] = Ab[max_row], Ab[k]
        #Computing mutipliers for each row from k+1
        for i in range(k+1, n):
            multiplier = Ab[i][k] / Ab[k][k]

            
            for j in range(k, n+1):
                Ab[i][j] = Ab[i][j] - multiplier * Ab[k][j]
        
    #Backward substitution
    x = [0]*n
    for i in range(n-1,-1,-1):
        x[i] = Ab[i][n]
        for j in range(i+1,n):
            x[i] = x[i] - Ab[i][j]*x[j]
        x[i] = x[i]/Ab[i][i] 
    return x


A = [
    [2, 1, 3, 0, 1],
    [1, 3, 2, 1, 0],
    [3, 2, 1, 1, 2],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 2, 3]
]

b = [
    2*1 + 1*2 + 3*3 + 0*4 + 1*5,    # = 2 + 2 + 9 + 0 + 5 = 18
    1*1 + 3*2 + 2*3 + 1*4 + 0*5,    # = 1 + 6 + 6 + 4 + 0 = 17
    3*1 + 2*2 + 1*3 + 1*4 + 2*5,    # = 3 + 4 + 3 + 4 + 10 = 24
    0*1 + 1*2 + 2*3 + 3*4 + 1*5,    # = 0 + 2 + 6 + 12 + 5 = 25
    1*1 + 0*2 + 1*3 + 2*4 + 3*5     # = 1 + 0 + 3 + 8 + 15 = 27
]


print(gemwec(A,b))