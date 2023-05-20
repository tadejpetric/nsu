import ProGED as pg
import numpy as np

def one_generation(eq, k, top):
    # higher k, higher learning rate. k in [0, infty)
    gramatika = eq.gen_grammar()
    generator = pg.GeneratorGrammar(gramatika)
    ED = pg.EqDisco(data=eq.data, 
                    lhs_vars=eq.lhs,
                    rhs_vars=eq.rhs,
                    generator=generator,
                    sample_size=500)
    ED.generate_models()
    ED.fit_models()
    print(ED.get_results())

    top.update(ED.get_results(top.n))

    E = np.array(eq.E)
    F = np.array(eq.F)
    T = np.array(eq.T)
    V = np.array(eq.V)

    for p in top.best:
        code = list(p.info["trees"].keys())[0]
        cr, cv = eq.parse(code)
        speed = (np.tanh(-np.log2(p.estimated["fun"])/k)+1)/2
        print(speed)
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

def evolve(eq, gens):
    top = leaderboard(3)
    for i in range(gens):
        E, F, T, V, top = one_generation(eq, 3, top) 
        print(E,F,T,V)
        eq.E = E
        eq.F = F
        eq.T = T
        eq.V = V
    return eq

if __name__ == "__main__":
    from make_data import *

    @pandise
    def first(x0, x1, x2, x3, x4):
        return x0 - 3*x1 - x2 -x4
    
    eq1 = problem(first)
    eq1.make_data(500)
    evolve(eq1, 50)