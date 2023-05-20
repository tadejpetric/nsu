import numpy as np
import pandas as pd
from functools import wraps
from inspect import signature
from nltk.tokenize import wordpunct_tokenize

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
        dataset[f"x{i}"] = np.random.default_rng(i).uniform(lower, upper, rows)
    
    dataset["y"] = function(dataset)
    return dataset

def get_rhs(function):
    columns = len(signature(function).parameters)
    return [f"x{i}" for i in range(columns)]

def tokenize(s):
    s = s.replace("+", "+ ")
    tokens_pre = wordpunct_tokenize(s)
    tokens2 = []
    
    prevpow = False
    for tk in tokens_pre:
        if prevpow:
            tokens2.append("**"+tk)
            prevpow = False
        elif tk != "**":
            tokens2.append(tk)
        else:
            prevpow = True

    tokens = []
    prevminus=False 
    for tk in tokens2:
        if prevminus:
            prevminus = False
            try:
                int(tk[0])
                tokens.append("-"+tk)
            except ValueError:
                tokens.append("-")
                tokens.append(tk)
        if tk == "-":
            prevminus = True
        else:
            tokens.append(tk)
    
    return tokens


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
        s += f"E -> E '+' F [{self.E[0]:0.16f}] | E '-' F [{self.E[1]:0.16f}] | F [{1-self.E[0]-self.E[1]:0.16f}]\n"
        s += f"F -> F '*' T [{self.F[0]:0.16f}] | F '/' T [{self.F[1]:0.16f}] | T [{1-self.F[0]-self.F[1]:0.16f}]\n"
        s += f"T -> V [{self.T[0]:0.16f}] | '(' E ')' [{self.T[1]:0.16f}] | 'sin(' E ')' [{1-self.T[0]-self.T[1]:0.16f}]\n"
        s += "V -> "

        total = 0
        # gramatika je od 0 do n-1
        for i, prob in enumerate(self.V):
            total += prob
            s += f"'x{i}' [{prob:0.16f}] | "
        
        s += f"'x{i+1}' [{1-total:0.16f}]"
        return s
    
    def parse(self, code : str):
        # we give higher weight to the binary symbols
        # similarly we have to increase weight for (E) and sin(E)
        count_rules = {"+":0, "-": 0, "F": 0, "*": 0, "/": 0, "T": 0, "V":0, "(":0, "sin":0}
        count_vars = {f"x{i}":0 for i in range(len(self.V)+1)}
        symbol_stack = ["E"]
        
        for ch in code:
            ch = int(ch)
            last = symbol_stack.pop()
            match last:
                case "E":
                    match ch:
                        case 0:
                            symbol_stack.extend(["F", "E"])
                            count_rules["+"] += 2
                        case 1:
                            symbol_stack.extend(["F", "E"])
                            count_rules["-"] += 2
                        case _:
                            symbol_stack.append("F")
                            count_rules["F"] += 1
                case "F":
                    match ch:
                        case 0:
                            symbol_stack.extend(["T", "F"])
                            count_rules["*"] += 2
                        case 1:
                            symbol_stack.extend(["T", "F"])
                            count_rules["/"] += 2
                        case _:
                            symbol_stack.append("T")
                            count_rules["T"] += 1
                case "T":
                    match ch:
                        case 0:
                            symbol_stack.append("V")
                            count_rules["V"] += 1
                        case 1:
                            symbol_stack.append("E")
                            count_rules["("] += 2
                        case _:
                            symbol_stack.append("E")
                            count_rules["sin"] += 2
                case "V":
                    count_vars[f"x{ch}"] += 1
        
        return (count_rules, count_vars)

        

class leaderboard:
    def __init__(self, n):
        self.best = []
        self.n = n
    
    def remove_dupes(self):
        # based on deduction tree. One could make it based on sympy expr instead
        temp = dict()
        for el in self.best:
            temp[list(el.info["trees"].keys())[0]] = el
        self.best = list(temp.values())


    def update(self, new):
        self.best.extend(new)
        self.remove_dupes()
        self.best = sorted(self.best, key=lambda e: e.estimated["fun"])[:self.n]