import numpy as np
import pandas as pd

def generiraj_enacbo_1(N=100):
    def f (x):
        return x[:,0] - 3*x[:,1] - x[:,2] + x[:,4]

    x = np.random.uniform(-10,10,N*5).reshape((N,5))
    y = f(x)
    d = {f"x{i+1}": x[:,i] for i in range(5)}
    d.update({"y":y})
    return pd.DataFrame(d)


def generiraj_enacbo_2(N=100):
    def f (x):
        return x[:,0]**5*x[:,1]**3

    x = np.random.uniform(-1,2,N*2).reshape((N,2))
    y = f(x)
    d = {f"x{i+1}": x[:,i] for i in range(2)}
    d.update({"y":y})
    return pd.DataFrame(d)

def generiraj_enacbo_3(N=100):
    def f (x):
        return np.sin(x[:,0]) + np.sin(x[:,1]/x[:,0]**2)

    x = np.vstack([np.random.uniform(0.5,1,N), np.random.uniform(0.5,3,N)]).T
    y = f(x)
    d = {f"x{i+1}": x[:,i] for i in range(2)}
    d.update({"y":y})
    return pd.DataFrame(d)

if __name__ == "__main__":
    data1 = generiraj_enacbo_1()
    print(data1.describe())

    data2 = generiraj_enacbo_2()
    print(data2.describe())

    data3 = generiraj_enacbo_3()
    print(data3.describe())
