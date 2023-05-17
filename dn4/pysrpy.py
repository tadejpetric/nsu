import numpy as np
import pandas as pd
from pysr import PySRRegressor

data = pd.read_csv("DN4_1_podatki.csv")
X = data.drop("Q", axis=1)
y = data[["Q"]]

model = PySRRegressor(
    niterations=200,  # < Increase me for better results
    #populations=20,
    #population_size=50,
    ## ^ Slightly larger populations, for greater diversity.
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
    #nested_constraints={
    #    "cos" : {"cos" : 0, "sin" : 0},
    #    "sin" : {"cos" : 0, "sin" : 0}
    #},
    ##timeout_in_seconds=60,
    #complexity_of_constants=0,
    weight_randomize=0.15,
    turbo=True,
    ##maxsize=20,
    maxdepth=10,
    procs=0,
    multithreading=False,
    random_state=0,
    deterministic=True
)

#model = PySRRegressor( #dela dobro!
#    niterations=40,  # < Increase me for better results
#    binary_operators=["+", "*", "-", "/"],
#    unary_operators=[
#        "cos",
#        "sin",
#        "square",
#        "cube",
#        "inv(x) = 1/x",
#        # ^ Custom operator (julia syntax)
#    ],
#    extra_sympy_mappings={"inv": lambda x: 1 / x},
#    # ^ Define operator for SymPy as well
#    # ^ Custom loss function (julia syntax)
#)

model.fit(X,y)

print("model", model)
print("score", model.score(X,y))


from matplotlib import pyplot as plt
plt.scatter(y, model.predict(X))
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()