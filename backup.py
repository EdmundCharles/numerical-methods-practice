import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline

# lnq = [4.37,4.49,4.61,4.74,4.83,4.92,4.98,5.06,5.15,5.24,5.32]
# lnT = [6.68,6.7,6.73,6.75,6.77,6.79,6.82,6.84,6.86,6.88,6.9]
# spline = UnivariateSpline(lnT,lnq,)

# plt.scatter(lnT,lnq)
# # print(*plt.xlim())
# xs = np.linspace(*plt.xlim())
# plt.xlim(plt.xlim())
# plt.plot(xs,spline(xs))
# plt.grid(True)
# plt.title('Рис.1 : График зависимости $ln(q)$ от $ln(T)$')
# plt.xlabel('$ln(T)$')
# plt.ylabel('$ln(q)$')
# plt.show()

f = lambda x: x**3-x**2-5*x -3
df = lambda x: 3*x**2 - 2*x -5
ddf = lambda x: 6*x -2
x = np.linspace(-3,3,1000)
plt.plot(x,f(x))
plt.plot(x,df(x))
plt.plot(x,ddf(x))
plt.show()