import numpy as np
import pandas as pd

from generation import evolve
import csv
from make_data import *

def write(function, sz, arr):
    with open(str(function)+str(sz)+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(arr)



@pandise
def first(x1, x2, x3, x4, x5):
    return x1 - 3*x2 - x3 -x5

@pandise
def second(x1, x2):
    return x1**5 * x2**3

@pandise
def third(x1, x2):
    return np.sin(x1) + np.sin(x2 / x1**2)

def dynamic(k):
    return lambda x: (np.tanh(-np.log2(x)/k)+1)/2

def adaptive_dynamic(start, end, slope):
    return lambda i: dynamic(max(-i/slope+start, end))

def constant(k):
    return lambda i: (lambda x: k)

def inv_pow(k):
    return lambda i: (lambda x: (1/(x+1))**k)

def fst(n):
    table = []
    for i in range(n):

        learning_rate = adaptive_dynamic(5,3,1)
        eq1 = problem(first)
        eq1.make_data(200, seed=i)

        np.random.seed(i)
        errors1 = evolve(eq1, learning_rate)
        x,y = zip(*errors1)
        table.append(y)
    write("first", 1000, table)

fst(30)