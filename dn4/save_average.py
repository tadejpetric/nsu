import numpy as np
import pandas as pd

from generation import evolve, dummy
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
    ssize = 1000
    gens = 50

    for i in range(n):

        learning_rate = adaptive_dynamic(5,3,1)
        eq1 = problem(first)
        eq1.make_data(200, seed=i)

        np.random.seed(i)
        errors1 = evolve(eq1, learning_rate, gens=gens, sample_size=1000)
        x,y = zip(*errors1)
        table.append(y)
        print("genetic fst", i)
    write("firstgen", ssize, table)

def snd(n):
    table = []
    ssize = 1000
    gens = 50

    for i in range(n):
        learning_rate = adaptive_dynamic(5,3,1)
        eq2 = problem(second)
        eq2.make_data(200, -1.8, 1.8, seed=i)
        np.random.seed(i)
        errors2 = evolve(eq2, learning_rate, gens=gens, sample_size=1000)
        x,y = zip(*errors2)
        table.append(y)
        print("genetic snd", i)

    write("sndgen", ssize, table)


def trd(n):
    table = []
    ssize = 300
    gens = 200

    for i in range(n):
        learning_rate = inv_pow(10)
        eq3 = problem(third)
        eq3.make_data(100, -1, -0.35, seed=i)
        eq3.data = pd.concat([eq3.data, makegraph(eq3._func, 100, 0.35, 2, seed=i)], ignore_index=True)
        #eq3.data.drop(eq3.data[abs(eq3.data.x0) < 0.3].index, inplace=True)

        np.random.seed(i)
        errors3 = evolve(eq3, learning_rate, gens=gens, sample_size=ssize)
        x,y = zip(*errors3)
        table.append(y)
        print("genetic trd", i)
        
    write("trdgen", ssize, table)

def fstdu(n):
    table = []
    ssize = 4000
    gens = 10
    for i in range(n):
        eq1 = problem(first)
        eq1.make_data(200, seed=i)

        np.random.seed(i)
        errors1 = dummy(eq1, gens=gens, sample_size=ssize)
        x,y = zip(*errors1)
        table.append(y)
        print("dummy fst", i)
    write("firstdum", ssize, table)

def snddu(n):
    table = []
    ssize = 4000
    gens = 10
    for i in range(n):
        eq2 = problem(second)
        eq2.make_data(200, -1.8, 1.8, seed=i)

        np.random.seed(i)
        errors1 = dummy(eq2, gens=gens, sample_size=ssize)
        x,y = zip(*errors1)
        table.append(y)
        print("dummy snd", i)
    write("seconddum", ssize, table)

def trddu(n):
    table = []
    ssize = 6000
    gens = 10
    for i in range(n):
        eq3 = problem(third)
        eq3.make_data(100, -1, -0.35, seed=i)
        eq3.data = pd.concat([eq3.data, makegraph(eq3._func, 100, 0.35, 2, seed=i)], ignore_index=True)
        #eq3.data.drop(eq3.data[abs(eq3.data.x0) < 0.3].index, inplace=True) # previous

        np.random.seed(i)
        errors1 = dummy(eq3, gens=gens, sample_size=ssize)
        x,y = zip(*errors1)
        table.append(y)
        print("dummy trd", i)
    write("thirddum", ssize, table)



#fst(10)
#fstdu(10)

print("--------first done---------")

#snd(10)
#snddu(10)

print("--------second done--------")

trd(10)
#trddu(10)