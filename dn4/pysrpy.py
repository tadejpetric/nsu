import numpy as np
import pandas as pd
from pysr import PySRRegressor

data = pd.read_csv("DN4_1_podatki.csv")
X = data.drop("Q", axis=1)
y = data[["Q"]]

model = PySRRegressor(
    niterations=3000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "cos",
        "sin",
        "square",
        "cube",
        "inv(x) = 1/x"
    ],
    denoise=True,
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    temp_equation_file="halls",
    tempdir="halls",
    delete_tempfiles=False,
    complexity_of_operators= {"sin": 3, "cos": 3, "cube": 4, "square": 2, "+" : 0.3, "*" : 0.3, "-": 0.3, "/" : 0.3},
    weight_randomize=0.15,
    turbo=True,
    maxdepth=10,
    procs=0,
    multithreading=False,
    random_state=0,
    deterministic=True
)


model.fit(X,y)

print("model", model)
print("score", model.score(X,y))


from matplotlib import pyplot as plt
plt.scatter(y, model.predict(X))
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
