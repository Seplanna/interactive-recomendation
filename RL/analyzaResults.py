import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def ResultsForOneUserEasy(user_res, utility, n_exp, result):
    length = 0
    u = 0
    n_ex = 0
    for i in user_res:
        if (i <= 0):
            length += 1
        else:
            m = min(len(result) - 1, length)
            result[m] += 1
            u += (0.5)**length
            n_ex += 1
            length = 0
    #if (length > 0):
    #    n_exp += 1
#    print(result)
    return utility + u/max(n_ex, 1), n_exp+1, result

def ResultsForOneUser(user_res, results, results_for_one_dim):
    n_zeros = 0
    used_res = set()
    n_i = 0
    for j, i in enumerate(user_res):
        if n_i >= len(results)-1:
            break
        if (i <= 0):
            results_for_one_dim[j] += 1
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
    mist = [0 for i in range(50)]
    results = [0 for i in range(dim+1)]
    results_for_dim = [0 for i in range(dim)]
    line_n = 0
    with open(file) as r:
        for line in r:
            if (line_n == 100):
               break
            #if (line_n < 300):
            #    line_n += 1
            #    continue
            line = line.strip().split()
            ResultsForOneUser([float(i) for i in line[start:]], results, results_for_dim)
            utility, n_exp, mist = ResultsForOneUserEasy([float(i) for i in line[start:]], utility, n_exp, mist)
            line_n += 1
    print(results[-1], results_for_dim[-1], utility / n_exp, n_exp)
    return [results, results_for_dim, utility / n_exp]

if __name__ == '__main__':

    n_q = 150
    n_st = -1
    for j in range(6, 7, 2):
        result_greedy = []
        utility_greedy = []
        result_our = []
        utility_our = []
        for i in range(0,n_q):
            print(i, j)
            r_g, r_d_g, u_g = Results('GreedyPLay_0', 50, i)
            r_o, r_d_o, u_o = Results('OurApproach3', 50, i)
            #print(j)
            #r_g, r_d_g, u_g = Results('results/' + str(j) +  '.txt', 20, i)
            #r_o, r_d_o, u_o = Results('results/' + str(j+2) +  '.txt', 20, i)
            result_greedy.append([r_g[n_st]])
            utility_greedy.append(u_g)
            result_our.append(r_o[n_st])
            utility_our.append(u_o)
        #print(result_greedy, result_greedy, utility_greedy, utility_our)

        a = range(n_q)
        # red dashes, blue squares and green triangles
        plt.plot(a, result_greedy, 'r', a, result_our, 'b')
        plt.show()
        plt.plot(a, utility_greedy, 'r', a, utility_our, 'b')
        plt.show()
    #plt.plot(a, result_greedy, 'r')
    #plt.show()
    #plt.plot(a, utility_greedy, 'r')
    #plt.show()