import numpy as np
import tangent
import matplotlib.pyplot as plt


def test_fun(x):
    out = x*x
    out = np.exp(-out)
    return out


df = tangent.grad(test_fun, verbose=1)

x = y = np.linspace(-2,2,100)
plt.plot(x,test_fun(x))
df_c = map(df, x)
plt.plot(x,list(df_c))
plt.show()
