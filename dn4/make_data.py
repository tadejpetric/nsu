import numpy as np
import pandas as pd
from functools import wraps
from inspect import signature

def variable(i):
    # doesn't work for like >25 variables
    return chr(ord('a')+i)

def pandise(function):
    # kinda proud of this decorator idea lol
    @wraps(function)
    def apply_to_table(tab):
        return function(*tab.to_numpy().transpose())
    
    return apply_to_table


def makegraph(function, rows, lower, upper):

    columns = len(signature(function).parameters)

    np.random.seed(1)
    dataset = pd.DataFrame()
    for i in range(columns):
        dataset[variable(i)] = np.random.default_rng(i).uniform(lower, upper, rows)
    
    dataset["y"] = function(dataset)
    return dataset

def get_rhs(function):
    columns = len(signature(function).parameters)
    return [str(variable(i)) for i in range(columns)]

class problem:
    def __init__(self, function):
        # assigns probabilities
        n = len(signature(function).parameters)

        self.E = [1/3, 1/3]
        self.F = [1/3, 1/3]
        self.T = [1/3, 1/3]
        self.V = [1/n]*(n-1)

        self._func = function
        self.rhs = get_rhs(function)
        self.lhs = ["y"]
    
    def make_data(self, rows, lower=-10, upper=10):
        self.data = makegraph(self._func, rows, lower, upper)


    def etoi(s):
        if "/": return 0
        if "-": return 1
        return 2
    def ftoi(s):
        if "*": return 0
        if "/": return 1
        return 2
    def ttoi(s):
        if "v": return 0
        if "e": return 1
        return 2
    def xtoi(s):
        return int(s[1:]) - 1
    def gen_grammar(self):
        s = ""
        s += f"E -> E '+' F [{self.E[0]}] | E '-' F [{self.E[1]}] | F [{1-self.E[0]-self.E[1]}]\n"
        s += f"F -> F '*' T [{self.F[0]}] | F '/' T [{self.F[1]}] | T [{1-self.F[0]-self.F[1]}]\n"
        s += f"T -> V [{self.T[0]}] | '(' E ')' [{self.T[1]}] | 'sin(' E ')' [{1-self.T[0]-self.T[1]}]\n"
        s += "V -> "

        total = 0
        # gramatika je od 0 do n-1
        for i, prob in enumerate(self.V):
            total += prob
            s += f"'{variable(i)}' [{prob}] | "
        
        s += f"'{variable(i+1)}' [{1-total}]"
        return s