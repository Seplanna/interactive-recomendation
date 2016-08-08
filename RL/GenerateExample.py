import random
import numpy as np

def GenerateData(dim, datadir):
    n_items = 4
    n_users = 3000
    items = np.random.rand(n_items, dim)
    items_bias = np.random.rand(n_items, 1)
    users = np.random.rand(n_users, dim)
    users_bias = np.random.rand(n_users, 1)
    np.savetxt(datadir + "/items.txt", items)
    np.savetxt(datadir + "/items_bias.txt", items_bias)
    np.savetxt(datadir + "/users.txt", users)
    np.savetxt(datadir + "/users_bias.txt", users_bias)

def Generate2DimExample(n_users, datadir):
    first_type = np.array([1., 1.])
    second_type = np.array([0., 1.])
    third_type = np.array([1., 0.])
    item_bias = np.array([2., 1., 1.])
    user1 = np.array([0.5, 0.5])
    user2 = np.array([1.5, 0.])
    user3 = np.array([0., 1.5])
    user4 = np.array([-2, 1])
    user5 = np.array([1., -2.])
    items = [first_type, second_type, third_type]
    users_type = [user1, user2, user3, user4, user5]
    user_bias = np.array([1., 1.5, 1.5, 2., 2.,])
    users= np.zeros([sum(n_users), 2])
    users_bias = np.zeros(sum(n_users))

    user_index = 0
    for i_,i in enumerate(n_users):
        for j in range(i):
            users[user_index] = users_type[i_]
            users_bias[user_index] = user_bias[i_]
            user_index += 1
    np.savetxt(datadir + "/items.txt", items)
    np.savetxt(datadir + "/items_bias.txt", item_bias)
    np.savetxt(datadir + "/users.txt", users)
    np.savetxt(datadir + "/users_bias.txt", users_bias)




if __name__ == '__main__':
#    GenerateData(3, "generateData")
    Generate2DimExample([0, 5, 5, 3, 3], "generateData")
