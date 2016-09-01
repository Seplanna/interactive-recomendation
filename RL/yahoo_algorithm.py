import sys
import numpy as np
from numpy.linalg import inv

def GetNextItem(items, used_items, l):
    u_i = list(used_items)
    if (len(u_i) < 1):
        p = np.zeros([1,items.shape[1]])
    else:
        p = items[u_i]
        p = np.vstack([p, np.zeros(p.shape[1])])
    ones = np.eye(p.shape[0])
    ones *= l
    best_res = 0
    best_item = 0
    for item in range(len(items)):
        if item not in used_items:
            p[-1] = items[item]
            res = np.matrix.trace(inv(np.dot(p, p.T) + ones))
            if res > best_res:
                best_res = res
                best_item = item
    return best_item

def MostInformativeItems(items, n_items):
    used_items = []
    for i in range(n_items):
        used_items.append(GetNextItem(items, used_items, 0.01))
    return used_items