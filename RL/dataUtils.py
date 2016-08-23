from random import gauss
import numpy as np

def SUCSESS():
    return 0.1


def recieveAnswer(user, user_bias, item, item_bias, global_bias, sigma):
    mean = np.dot(item, user)
    return mean + user_bias + item_bias + global_bias

def GetSimilarUsers(user, user_bias, users_estimation, user_bias_estimation):
    n_users = users_estimation.shape[0]
    distance_to_users = [[np.dot(user - users_estimation[i], (user - users_estimation[i]).T) +
                          (user_bias - user_bias_estimation[i])**2, i]
                          for i in range(n_users)]
    #distance_to_users = [[np.dot(user, users_estimation[i].T) +
    #                      (user_bias - user_bias_estimation[i]) ** 2, i]
    #                     for i in range(n_users)]
    distance_to_users.sort(key=lambda x:x[0])
    return distance_to_users

def GetReward(distance_to_users, users, users_bias, item, item_bias, global_bias, sigma):
    n_users = 0
    result = 0
    i = 0
    while i < users.shape[0] and distance_to_users[i][0] < 0.01:
        n_users += 1
        result += int(recieveAnswer(users[distance_to_users[i][1]], users_bias[distance_to_users[i][1]],
                                   item, item_bias, global_bias, sigma)
                     > SUCSESS())
        i += 1
    #print('n_users = ', n_users)
    return result / float(n_users)

def GetData(datadir):
    item_vecs = np.genfromtxt(datadir + "/items.txt")
    item_bias = np.genfromtxt(datadir + "/items_bias.txt")
    user_vecs = np.genfromtxt(datadir + "/users.txt")
    user_bias = np.genfromtxt(datadir + "/user_bias.txt")
    global_bias = 0.
    with open(datadir + "/global_bias.txt", 'r') as global_b:
        for line in global_b:
            global_bias = float(line.strip())
    return item_vecs, item_bias, user_vecs, user_bias, global_bias

def ExpandData(user_vecs, user_bias, expand):
    n_users = user_vecs.shape[0]
    new_user_vecs = np.empty([expand * n_users, user_vecs.shape[1]])
    new_user_bias = np.empty(expand * n_users)
    for i in range(n_users):
        for j in range(expand):
            new_user_vecs[i * expand + j] = user_vecs[i]
            new_user_bias[i * expand + j] = user_bias[i]
    return [new_user_vecs, new_user_bias]


def appendAllArrays(arrays):
    c = np.array([])
    for ar in arrays:
        c = np.append(c, ar)
    return c


def make_input(item_vec, item_bias, user_vec, user_bias):
    return appendAllArrays([item_vec, item_bias, user_vec, user_bias, item_vec * user_vec])
    #return item_vec * user_vec




def OneStep(user, user_bias, item, item_bias, global_bias, r, learning_rate, user_bias_reg, user_fact_reg):
    prediction = global_bias + user_bias + item_bias
    prediction += np.dot(user, item)
    e = (r - prediction)  # error

    # Update biases
    user_bias += learning_rate * \
                         (e - user_bias_reg * user_bias)
    # Update latent factors
    user += learning_rate * \
                            (e * item - \
                             user_fact_reg * user)
    return [user, user_bias]

def GetBestItem(item_vecs, item_bias, user, user_bias, W, used_items):
    y1_arg = 0
    element = -1000
    for item1 in range(item_vecs.shape[0]):
        c = np.dot(W, make_input(item_vecs[item1], item_bias[item1],
                                                        user, user_bias))
        if (element < c and not item1 in used_items):
            element = c
            y1_arg = item1
    return y1_arg

def SortIItemByPopularity(users, users_bias, item, item_bias, global_bias, sigma):
    result = []
    n_users = users.shape[0]
    for i in range(item.shape[0]):
        result.append([
            sum(float(recieveAnswer(users[j], users_bias[j], item[i], item_bias[i], global_bias, sigma) > SUCSESS()) for j in range(users.shape[0])) / n_users ,
            item_bias[i] , i])
    result.sort(key = lambda x:x[0])
    return result

def learning_rate(step):
    return 1./math.sqrt(step)

def GetItemsNames(file):
    res = {}
    with open(file) as f:
        for line in f:
            line = line.split('|')
            res[int(line[0])] = line[1]
    return res

def PrintItemPopularity(user_vecs, user_bias, item_vecs, item_bias, global_bias, sigma):
    sort_items = SortIItemByPopularity(user_vecs, user_bias, item_vecs, item_bias, global_bias, sigma)
    items_names = GetItemsNames("data/u.item")
    with open("itemPopularity", 'w') as ip:
        for i in sort_items:
            print(i[-1], items_names[i[-1] + 1])
            ip.write(str(i[0]) + "\t"  + str(i[1]) + "\t"  + str(i[2]) +  "\t" + items_names[i[2] + 1] + '\n')

def GetItemPopularity():
    result = []
    with open("itemPopularity", 'r') as ip:
        for line in ip:
            line = int(line.strip().split("\t")[2])
            result.append(line)
    return list(reversed(result))

def DeleteMostPopularItems(n_delete_items):
    items_pop = GetItemPopularity()
    with open("data/items.txt") as items, \
         open("data/items_bias.txt") as i_b, \
         open("data/items1.txt", 'w') as n_items,\
         open("data/items_bias1.txt", 'w') as n_ib:
        n_line = 0
        for line in items:
            if n_line not in items_pop[:n_delete_items]:
                n_items.write(line)
            n_line += 1
        n_line = 0
        for line in i_b:
            if n_line not in items_pop[:n_delete_items]:
                n_ib.write(line)
            n_line += 1

def VectorToString(vector):
    return "_".join(str(v) for v in vector)

def LearningRate(learning_rate, step):
    res = learning_rate / step
    if (res < 0.01):
        res = 0.01
    return res


class StateStat(object):
    def __init__(self):
        self.nq = 0
        self.npositiv = 0