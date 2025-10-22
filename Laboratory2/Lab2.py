import numpy as np
import matplotlib.pyplot as plt

def print_matrix(matrix):
    for row in matrix:
        print(row)
def GEM_MOD(A, bt):
    """
    This function implements the Modified Gaussian elimination method for solving systems of linear equations Ax=b

    Parameters
    ----------
    A : array_like
        (n, n) array representing the coefficient matrix.
    bt : array_like
        (1,n) array representing the transposed free term vector.

    ReturnsM
    -------
    x : array
        The solution vector for the given system of linear equations.
    """
    #Preparation
    n = len(A)
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError('The matrix contains zeros in the diagonal')
    augmented_matrix = [A[i]+[bt[i]] for i in range(n)]
    Ab = augmented_matrix

    #Forward elimination
    for k in range(n): 
        #Finding the largest A[i][k] and swapping the rows
        max_val = abs(Ab[k][k])
        max_row = k
        for i in range(k+1,n):
            if abs(Ab[i][k]) > max_val:
                max_val = abs(Ab[i][k])
                max_row = i
        if max_row != k:
            Ab[k], Ab[max_row] = Ab[max_row], Ab[k]
        
        #Computing mutipliers for each row from k+1
        for i in range(k+1, n):
            multiplier = Ab[i][k] / Ab[k][k]

            for j in range(k, n+1):#Вынести в отлельный метод вычитание векторов
                Ab[i][j] = Ab[i][j] - multiplier * Ab[k][j]
            
        
    #Backward substitution
    x = [0]*n
    for i in range(n-1,-1,-1):
        x[i] = Ab[i][n]
        for j in range(i+1,n):#Отлельно реализоаать через сумму какую нибудь, по сути это скалярное произведение i строки на вектор с уже известными иксами
            x[i] = x[i] - Ab[i][j]*x[j]
        x[i] = x[i]/Ab[i][i]    
    for i in range(len(x)):
        x[i] = float(x[i])
    return x


