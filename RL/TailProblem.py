import sys
import numpy as np
from q_learning import*

def AskPopularItems(user, user_b, items, items_bias, items_n, items_popul, global_bias,learning_rate, sigma):
    user_estim = np.zeros(user.shape[0])
    user_b_estim = 0
    for i in range(items_n):
        item = items[items_popul[i]]
        item_b = items_bias[items_popul[i]]
        reward1 = recieveAnswer(user, user_b, item, item_b, global_bias, sigma)
        user_estim, user_b_estim = \
            OneStep(user_estim, user_b_estim, item, item_b, global_bias, reward1, learning_rate,
                    learning_rate, learning_rate)
        return user_estim, user_b_estim

def newItems(items, items_b, item_pop, items_n):
    items_not_included = set(item_pop[i] for i in range(items_n))
    new_items = np.zeros((items.shape[0] - items_n, items.shape[1]))
    new_items_b = np.zeros(items.shape[0]- items_n)
    index = 0
    for i in range(items.shape[0]):
        if i not in items_not_included:
            new_items[index] = items[i]
            new_items_b[index] = items_b[i]
            index += 1
    np.savetxt("data1/items", new_items)
    np.savetxt("data1/items_bias", new_items_b)


def main():
    items_popul = GetItemPopularity()
    sigma = 0.5
    eps = 0.1
    learning_rate = 0.1
    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    user_estimation = np.zeros((user_vecs.shape[0], user_vecs.shape[1]))
    user_bias_estim = np.zeros(user_vecs.shape[0])
    for i in range(user_vecs.shape[0]):
        user_estimation[i], user_bias_estim[i] = AskPopularItems(user_vecs[i],
                                     user_bias[i],
                                     item_vecs,
                                     item_bias,
                                     10,
                                     items_popul,
                                     global_bias,
                                     learning_rate,
                                     sigma)
    np.savetxt("data1/users", user_estimation)
    np.savetxt("data1/users_bias", user_bias_estim)
    newItems(item_vecs, item_bias, items_popul, 10)

main()