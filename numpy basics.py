import numpy as np
import matplotlib.pyplot as plt

lol = np.array([1, 2, 3, 4])
print(lol ** 2)
print(np.arange(10) ** 2)
print(type(lol))
print(lol.dtype)
print(lol.shape)
lol = lol.reshape(2, 2)
print(lol)

ok = np.arange(3*3).reshape(3, 3)
print(ok.shape)
print(ok[1, 2])
fok = np.arange(10, dtype='f')
print(fok.dtype.itemsize)
print(fok.mean())
print(fok.var())
print(fok.T)
print(fok[len(fok)::-1])
print(np.linspace(9,10).size)
neo = np.arange(5,50,5)
print(np.where(neo%15==0))
print(neo[np.where(neo%15==0)])
print(neo[neo%15==0])


x = np.random.normal(0,1,100)
y = np.random.normal(0,1,100)
fp1 = np.polyfit(x,y,2)
fx = np.poly1d(fp1)
print(fx)
y = np.random.normal(0,1,100)
plt.scatter(x,y)
plt.plot(x, fx(x), 'green')
plt.show()

