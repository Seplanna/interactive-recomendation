import sys
def ResultsForOneUserEasy(user_res, utility, n_exp):
    length = 0
    u = 0
    n_ex = 0
    for i in user_res:
        if (i == 0):
            length += 1
        else:
            u += (0.1)**length
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
        if (i == 0):
            n_zeros += 1
            results[-1] += 1
            #if True :
            if (n_zeros - 1 not in used_res):
                results[n_zeros - 1] += 1
                used_res.add(n_zeros - 1)
        else:
            n_zeros = 0
        n_i += 1

def Results(file, dim):
    utility = 0
    n_exp = 0
    results = [0 for i in range(dim+1)]
    line_n = 0
    with open(file) as r:
        for line in r:
            #if (line_n == 100):
            #   break
            #if (line_n < 100):
            #    line_n += 1
            #    continue
            line = line.strip().split()
            ResultsForOneUser([int(i) for i in line], results)
            utility, n_exp = ResultsForOneUserEasy([int(i) for i in line], utility, n_exp)
            line_n += 1
    print(results, utility / n_exp, n_exp)

if __name__ == '__main__':
    Results('GreedyPlay', 20)
    Results('OurApproach3', 20)