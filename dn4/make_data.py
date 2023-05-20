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
    
    def parse(self, s : str):
        # every parsing of E concludes with the third rule. If + was enacted once, F was then enacted in addition
        tokens = tokenize(s)
        count_rules = {"+":0, "-": 0, "F": 0, "*": 0, "/": 0, "T": 0, "V":0, "(":0, "sin":0}
        count_vars = {f"x{i}":0 for i in range(len(self.V)+1)}
        was_sin = False
        no_E_op = True
        no_F_op = True

        
        for i,token in enumerate(tokens):
            if token[:2] == "**":
                no_F_op = False
                z = int(token[2:])-1
                count_rules["*"] += z*2
                count_rules["T"] += z
                if tokens[i-1][0] == "x":
                    count_vars[tokens[i-1]] += z
                continue
            if was_sin:
                # if previous token was "sin", next token is "("
                was_sin = False
                continue
            try:
                # cases like 3(x1+x2) (not 3x1), artifact of parsing
                z = int(token)-1
                count_rules["+"] += z*2
                count_rules["F"] += z
                no_E_op = False
                continue
            except ValueError:
                pass
            if token[0]=="x":
                # every xn came from a V
                count_vars[token] += 1
                count_rules["V"] += 1
                count_rules["F"] += no_E_op
                count_rules["T"] += no_F_op

            elif "x" in token:
                # cases like x2   +x1+x1 -> x2+  2x1 

                left, right = token.split("x")
                if left[0] == "-":
                    # -2x1 = -x1-x1
                    # technically -2x1 = x1-x1-x1-x1
                    # but since x2 -2x1 = x2 -x1 -x1, we choose this route for simplicity
                    left = int(left[1:])
                    no_E_op = False
                    count_rules["-"] += left*2
                    count_rules["F"] += left
                    count_vars["x"+right] += left
                else:
                    left = int(left)
                    no_E_op = False
                    count_rules["+"] += left*2
                    count_rules["F"] += left
                    count_vars["x"+right] += left
            elif token == "+":
                count_rules["+"] += 2
                count_rules["F"] += 1
                no_E_op = False
            elif token == "-":
                count_rules["-"] += 2
                count_rules["F"] += 1
                no_E_op = False 
        

            elif token == "sin":
                count_rules["sin"] += 1
                count_rules["F"] += no_E_op
                count_rules["T"] += no_F_op
                was_sin = True
                no_F_op = True
                no_E_op = True

            elif token == "(":
                count_rules["("] += 1
                count_rules["F"] += no_E_op
                count_rules["T"] += no_F_op
                no_F_op = True
                no_E_op = True
            elif token == "*":
                try:
                    # 3*x1
                    z = int(tokens[i-1])
                    if i-1 > 0:
                        if tokens[i-2] == "-":
                            z *= -1
                    if z > 0:
                        count_rules["+"] += 2*z
                        count_rules["F"] += z
                        if tokens[i+1][0] == "x":
                            count_vars[tokens[i+1]] += z
                    else:
                        z *= -1
                        count_rules["-"] += 2*z
                        count_rules["F"] += z
                        if tokens[i+1][0] == "x":
                            count_vars[tokens[i+1]] += z
                except ValueError:
                    count_rules["*"] += 2
                    count_rules["T"] += 1
                no_F_op = False
            elif token == "/":
                count_rules["/"] += 2
                count_rules["T"] += 1
                no_F_op = False

        count_rules["F"] += no_E_op
        count_rules["T"] += no_F_op
        if tokens[0][0] == "-" and count_rules["+"] > 1:
            count_rules["+"] -= 2
        return (count_rules, count_vars)





