import numpy as np
from GEM import GEM_MOD
import scipy.linalg as sc
#Code from class 09.10.25
#In lab2 n shall not be less than 10
n = 10

a = np.random.rand(n,n)
a = a.tolist()
# a =  sc.hilbert(n) #------> huge error cuz hilbert
x = [i for i in range(1,n+1)]

b = np.matmul(a,x)
b = b.tolist()
# print(b)
x_num = np.linalg.solve(a,b)

err_x = np.linalg.norm(x - x_num)/np.linalg.norm(x)
print(f'error = {err_x}')
cond = np.linalg.cond(a)
print(f'condition number of the matrix = {cond}')

# print(gemwec())

#generating a matrix with a specified cond 

# cond_desired = 15
# l = np.linspace(1,cond_desired,n)
# d = np.diag(l)
# q = np.linalg.qr(np.random.rand(n,n)).Q

# a_cond = np.matmul(np.matmul(q,d),np.linalg.matrix_transpose(q))
# print(np.linalg.cond(a_cond))

print(GEM_MOD(a,b))