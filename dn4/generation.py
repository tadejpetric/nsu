import ProGED as pg
import numpy as np
from make_data import *
import warnings

def one_generation(eq, f_err, size, top):
    
    # higher k, higher learning rate. k in [0, infty)
    gramatika = eq.gen_grammar()
    generator = pg.GeneratorGrammar(gramatika)
    ED = pg.EqDisco(data=eq.data, 
                    lhs_vars=eq.lhs,
                    rhs_vars=eq.rhs,
                    generator=generator,
                    sample_size=size)
    ED.generate_models()
    ED.fit_models()
    print(ED.get_results())

    top.update(ED.get_results(top.n))

    E = np.array(eq.E)
    F = np.array(eq.F)
    T = np.array(eq.T)
    V = np.array(eq.V)

    for p in top.best[::-1]:
        code = list(p.info["trees"].keys())[0]
        cr, cv = eq.parse(code)

        speed = f_err(p.estimated["fun"])
        #print(speed)
        # "+":0, "-": 0, "F": 0,
        total = cr["+"] + cr["-"] + cr["F"]
        # desired is [cr["+"], cr["-"]] / total
        E2 = np.array([cr["+"], cr["-"]])/total
        total = cr["*"] + cr["/"] + cr["T"]
        F2 = np.array([cr["*"], cr["/"]])/total
        total = cr["V"] + cr["("] + cr["V"]
        T2 = np.array([cr["V"], cr["("]])/total
        total = sum(cv.values())
        del cv[f"x{len(V)}"]
        V2 = np.array(list(cv.values()))/total

        E = E*(1-speed) + E2*speed
        F = F*(1-speed) + F2*speed
        T = T*(1-speed) + T2*speed
        V = V*(1-speed) + V2*speed
    
    return (list(E),list(F),list(T),list(V), top)

def evolve(eq, k, gens=50, sample_size=1000,  lead_size=4):
    warnings.filterwarnings("ignore")
    top = leaderboard(lead_size)
    errors = []
    for i in range(gens):
        E, F, T, V, top = one_generation(eq, k(i), sample_size, top)
        errors.append((i*sample_size, top.best[0].estimated["fun"]))
        print(E,F,T,V)
        eq.E = E
        eq.F = F
        eq.T = T
        eq.V = V

        if top.best[0].estimated["fun"] < 1e-6:
            break
    return errors

def dummy(eq, gens=50, sample_size=1000):
    top = leaderboard(1)
    errors = []
    gramatika = eq.gen_grammar()
    generator = pg.GeneratorGrammar(gramatika)
    for i in range(gens):
        ED = pg.EqDisco(data=eq.data, 
                        lhs_vars=eq.lhs,
                        rhs_vars=eq.rhs,
                        generator=generator,
                        sample_size=sample_size)
        ED.generate_models()
        ED.fit_models()
        print(ED.get_results())
        top.update(ED.get_results(top.n))
        errors.append((i*sample_size, top.best[0].estimated["fun"]))
        if top.best[0].estimated["fun"] < 1e-6:
            break
    return errors


if __name__ == "__main__":
    from make_data import *

    @pandise
    def first(x0, x1, x2, x3, x4):
        return x0 - 3*x1 - x2 -x4
    
    @pandise
    def second(x1, x2):
        return x1**5 * x2**3

    @pandise
    def third(x1, x2):
        return np.sin(x1) + np.sin(x2 / x1**2)
    
    np.random.seed(3)
    learning_rate = lambda i: min(-i+5, 3)

    eq1 = problem(first)
    eq1.make_data(200)
    errors = evolve(eq1, learning_rate)
    x,y = zip(*errors)