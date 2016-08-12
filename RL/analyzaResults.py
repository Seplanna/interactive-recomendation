import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def ResultsForOneUserEasy(user_res, utility, n_exp):
    length = 0
    u = 0
    n_ex = 0
    for i in user_res:
        if (i <= 0):
            length += 1
        else:
            u += (0.5)**length
            n_ex += 1
            length = 0
    #if (length > 0):
    #    n_exp += 1
    return utility + u/max(n_ex, 1), n_exp+1

def ResultsForOneUser(user_res, results):
    n_zeros = 0
    used_res = set()
    n_i = 0
    for i in user_res:
        if n_i >= len(results)-1:
            break
        if (i <= 0):
            n_zeros += 1
            results[-1] += 1
            #if True :
            if (n_zeros - 1 not in used_res):
                results[n_zeros - 1] += 1
                used_res.add(n_zeros - 1)
        else:
            n_zeros = 0
        n_i += 1

def Results(file, dim, start):
    utility = 0
    n_exp = 0
    results = [0 for i in range(dim+1)]
    line_n = 0
    with open(file) as r:
        for line in r:
            if (line_n == 50):
               break
            #if (line_n < 300):
            #    line_n += 1
            #    continue
            line = line.strip().split()
            ResultsForOneUser([float(i) for i in line[start:]], results)
            utility, n_exp = ResultsForOneUserEasy([float(i) for i in line[start:]], utility, n_exp)
            line_n += 1
    print(results, utility / n_exp, n_exp)
    return [results, utility / n_exp]

if __name__ == '__main__':
    result_greedy = []
    utility_greedy = []
    result_our = []
    utility_our = []

    n_q = 10
    n_st = -1
    for i in range(0,n_q):
        print(i)
        r_g, u_g = Results('GreedyPlay_1', 30, i)
        #r_o, u_o = Results('OurApproach3', 30, i)
        result_greedy.append([r_g[n_st]])
        utility_greedy.append(u_g)
        #result_our.append(r_o[n_st])
        #utility_our.append(u_o)
        print(result_greedy, result_greedy, utility_greedy, utility_our)

    a = range(n_q)
    # red dashes, blue squares and green triangles
    #plt.plot(a, result_greedy, 'r', a, result_our, 'b')
    #plt.show()
    #plt.plot(a, utility_greedy, 'r', a, utility_our, 'b')
    #plt.show()
    plt.plot(a, result_greedy, 'r')
    plt.show()
    plt.plot(a, utility_greedy, 'r')
    plt.show()