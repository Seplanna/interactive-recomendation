from random import gauss
from random import randint
import math
from logistic_regression import *
from dataUtils import *
from Einviirenment import *
import matplotlib.pyplot as plt


def Test(env, classifiers, n_q, n_it, distancse_file):
    states_item_policy = {}
    f = open("results/" + str(n_it) + ".txt", 'w')
    n_min = 0
    distanse_to_real = np.zeros(n_q)
    parameters = First_approximation(env.latent_dim)
    items = []
    learning_rate = 0.1
    for i1 in range(env.n_users / 4):
        items = []
        #if (i1 == 20):
        #    break
        i  = 4*i1
        #i = i1
        users_used_items = set()
        env.user_vecs_estim[i] = np.zeros(env.latent_dim)
        env.user_bias_estim[i] = 0
        rewards = [0 for q in range(n_q)]

        for j in range(n_q):
            c = min(len(classifiers) - 1, j)
            a = classifiers[c]

            user = np.append(env.user_vecs_estim[i], env.user_bias_estim[i])
            user_str = VectorToString(user)
            if user_str in states_item_policy:
                item = states_item_policy[user_str]
            else:
                max_q, best_item, item = a.recieve_new_greedy_action(env.actions, user, users_used_items)
                states_item_policy[user_str] = item
            reward = env.reward(i, item)
            items.append(item)
            r = -1.
            n_min += 1
            if (reward > SUCSESS()):
                r = 1.
                n_min -= 1
            users_used_items.add(item)
            rewards[j] = r
            env.update_state(i, item, LearningRate(learning_rate, j+1)  )
            d = env.user_vecs_estim[i] - env.user_vecs[i]
            d = np.dot(d, d.T)
            distanse_to_real[j] += math.sqrt(d)
        if i%200 == 0 and i > 0:
            print(i)
            print(distanse_to_real)
            break
        f.write("\t".join(str(rew) for rew in rewards) + '\n')
        #print(rewards)
    f.close()
    distanse_to_real /= (env.n_users / 4)
    a = range(n_q)
   # plt.plot(a, np.array(distanse_to_real) , 'r')
   # plt.show()
    print(distanse_to_real)
    distancse_file.write("\t".join(str(d) for d in distanse_to_real) + '\n')
    print(n_min)
    print(items)

def Get_classifiers():
    n_classifiers = 10
    classifiers = []
    for i in range(n_classifiers):
        classifiers.append(Qlearning(first_W = np.genfromtxt("parameters4_" + str(i))))

    n_q = 200
    expand = 1
    learning_rate = 0.01
    n_users = 100
    sigma = 0.5
    env = Envierment(expand, n_users + 100, sigma, learning_rate)
    Test(env, classifiers, n_q, 100, "results/distance_1.txt")


def Learn():
    distancse_file = open('results/distance.txt', 'w')

    n_classifiers = 10
    expand = 4
    learning_rate = 0.1

    learning_rate1s = []
    for i in range(n_classifiers):
        learning_rate1s.append(th.shared(0.05))

    batch_size = 100
    n_users = 400
    sigma = 0.5
    env = Envierment(expand, n_users, sigma, learning_rate)

    r = T.dscalar('r')
    z = T.dscalar('z')
    lambd = T.dscalar('lambd')
    state = T.dvector('state')
    action = T.dvector('action')
    new_state = T.dvector('new_state')
    new_action = T.dvector('new_action')

#parameters in  classifier
    parameters = []
    W1s = []
    classifiers = []
    for i in range(n_classifiers):
        parameters.append(First_approximation(env.latent_dim))
        W1s.append(th.shared(parameters[-1]))
        classifiers.append(Qlearning(first_W = parameters[-1]))
    #parameters = First_approximation(env.latent_dim)
    #W1 = th.shared(parameters)
    #classifier = Qlearning(first_W=parameters)

#functions
    costs = []
    predictions = []
    new_predictions = []
    for i in range(n_classifiers):
        costs.append((classifiers[i].q(state, action) - r  -  lambd * classifiers[i].q(new_state, new_action))**2)
        predictions.append(classifiers[i].q(state, action))
        new_predictions.append(classifiers[i].q(new_state, new_action))

#updates
    f_updates = []
    fs = []
    g_Ws = []
    updates = []
    updates1s = []
    f_ws = []
    for i in range(n_classifiers):
        f_updates.append([(learning_rate1s[i], 0.05 / (1. / (20 * learning_rate1s[i]) + 1))])
        fs.append(th.function(inputs=[z], outputs=z * learning_rate1s[i], updates=f_updates[-1]))
        g_Ws.append(T.grad(cost=costs[i], wrt=classifiers[i].log_regression.W))
        updates.append([(W1s[i], W1s[i] - learning_rate1s[i] * (1. / batch_size) * g_Ws[-1])])
        updates1s.append([(classifiers[i].log_regression.W, W1s[i])])
        f_ws.append(th.function(inputs=[z], outputs=z, updates=updates1s[-1]))


    #f_updates = [(learning_rate1, 0.05 / (1. / (20 * learning_rate1) + 1))]
    #f = th.function(inputs=[z], outputs=z * learning_rate1, updates=f_updates)
    #g_W = T.grad(cost=cost, wrt=classifier.log_regression.W)
    #updates = [(W1, W1 - learning_rate1 * (1. / batch_size) * g_W)]
    #updates1 = [(classifier.log_regression.W, W1)]
    #f_w = th.function(inputs=[z], outputs=z, updates=updates1)

#train_function
    trains = []
    for i in range(n_classifiers):
        trains.append (th.function(
            inputs=[state, action, new_state, new_action, r, lambd],
            outputs=[predictions[i], new_predictions[i], costs[i]],
            updates=updates[i]))

#should be changed
    users_used_items = [set() for i in range(n_users)]

    i1 = 0
    while i1 < 40:
        print(i1)
        if (i1 > 5 and i1 % 2 == 0):
            Test(env, classifiers, 40, i1, distancse_file)
        n_u = min(i1, n_classifiers)
        for i in range(n_u):
            f_ws[i](1.)
            print(classifiers[i].log_regression.W.get_value())
        states_item_policy = {}
        for i2 in range(env.n_users):
            i = randint(0, env.n_users - 1)
            if (i2 % batch_size == 0):
                for ii in range(n_u):
                    f_ws[ii](1.)
                    print(i1)
                    #print(classifier.log_regression.W.get_value())
                    print("learning rate ", fs[ii](1))
                    np.savetxt("parameters4_" + str(ii), classifiers[ii].log_regression.W.get_value(borrow=True))
            users_used_items[i] = set()
            env.user_vecs_estim[i] = np.zeros(env.latent_dim)
            env.user_bias_estim[i] = 0
            rewards = [0 for s in range(i1 + 1)]
            train_parameters = []
            for i3 in range(i1 + 1):
                user = np.append(env.user_vecs_estim[i], env.user_bias_estim[i])
                user_str = VectorToString(user)
                if user_str in states_item_policy:
                    item = states_item_policy[user_str]
                else:
                    c_n = min(i3, n_classifiers - 1)
                    a = classifiers[c_n]
                    max_q, best_item, item = a.recieve_new_greedy_action(env.actions, user, users_used_items[i])
                    states_item_policy[user_str] = item

                user_real = np.append(env.user_vecs[i], env.user_bias[i])
                max_q_1, best_item_1, item_1 = a.recieve_new_greedy_action(env.actions, user_real, users_used_items[i])
                train_parameters.append([user, best_item, item, user_real, best_item_1, item_1])
                reward = env.reward(i, item)
                r = -1.
                if (reward > SUCSESS()):
                    r = 1.
                users_used_items[i].add(item)
                old_dis = np.dot((user - user_real), (user-user_real).T)
                env.update_state(i, item, LearningRate(learning_rate, i3+1) )
                new_user = np.append(env.user_vecs_estim[i], env.user_bias_estim[i])
                new_dis = np.dot((new_user - user_real), (new_user - user_real).T)
                rewards[i3] = 10 * (-new_dis + old_dis) + r

            new_user = np.append(env.user_vecs_estim[i], env.user_bias_estim[i])
            new_max_q, new_best_item, new_item = a.recieve_new_greedy_action(env.actions, user, users_used_items[i])
            r = 0
            l = 1
            l1 = 1
            for i3 in range(i1 +1):
                cl =  i1 - i3  - 1
                if (rewards[-i3-1] < -1000):
                    #l *= 0.33
                    #r*= 0.33
                    r -= 0
                    if (l1 < l):
                        l1 = l
                else:
                    #l *= 0.95
                    #r = 0.5 + r * 0.95
                    r += rewards[-i3-1]
                if cl >= n_classifiers:
                    continue

                pred, new_prediction, err = trains[cl](train_parameters[-i3 - 1][0], train_parameters[-i3 - 1][1],
                                                       new_user, new_best_item,
                                                       r, 1.)

            if (i2 % batch_size == 0):
                print(i2)
                items = [it[2] for it in train_parameters]
                print("items", items)
                print(rewards)
                print("Pred, err, reward, after", pred, err, (float(r) / (2 * (i3 + 1))), new_prediction)
        for i in range(n_classifiers):
            print("learning rate ", fs[i](1))
        i1 += 1
        for i in range(n_classifiers):
            np.savetxt("parameters4_" + str(i), classifiers[i].log_regression.W.get_value(borrow=True))
    distancse_file.close()

def main1():
    learning_rate = 0.001
    learning_rate1 = th.shared(0.05)
    sigma = 0.5
    eps = 0.1


    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    latent_dim = user_vecs.shape[1]
    #n_users = user_vecs.shape[0]
    expand = 4
    n_users = 400
    user_vecs, user_bias = ExpandData( user_vecs, user_bias, expand)
    x = T.dvector('x')
    y = T.dvector('y')
    z = T.dscalar('z')
    r = T.dscalar('r')

    W = np.zeros(3 * latent_dim + 2)
    W[latent_dim] = 1.
    for i in range(latent_dim):
        W[-i-1] = 1.
#    W = np.genfromtxt("parameters")
    W1 = th.shared(W)
    classifier = Qlearning(first_W=W)


    f_updates = [(learning_rate1, 0.05 / T.sqrt((1. / ((20 * learning_rate1) ** 2)) + 1))]
    f = th.function(inputs = [z], outputs = z * learning_rate1, updates = f_updates)
    cost = (classifier.cost(x) - r - (1 - r) * 0 * classifier.cost(y)) ** 2
    g_W = T.grad(cost=cost, wrt=classifier.W)

    batch_size = 399
    updates = [(W1, W1 - learning_rate1 * (1./batch_size) * g_W)]
    updates1 = [(classifier.W, W1)]
    f_w = th.function(inputs = [z], outputs = z, updates = updates1)

    #item_vecs, item_bias, user_estimation, user_bias_estim, global_bias1 = GetData("data1")
    #user_estimation, user_bias_estim = ExpandData(user_estimation, user_bias_estim, expand)
    user_estimation = np.zeros((user_vecs.shape[0], user_vecs.shape[1]))
    user_bias_estim = np.zeros(user_vecs.shape[0])
    #n_questions = np.zeros(user_vecs.shape[0])
    prediction = classifier.cost(x)
    new_prediction = classifier.cost(y)

    train = th.function(
        inputs=[x, y, r],
        outputs=[prediction, new_prediction, cost],
        updates= updates)
    i1 = 0
    users_used_items = [set() for i in range(user_vecs.shape[0])]

    while i1 < 20:
        f_w(1.)
        print (W1.get_value())
        print(classifier.W.get_value())
        for i2 in range(n_users):
            i = randint(0, n_users)
            new_item = 1
            if (i2 % batch_size == 0):
                f_w(1.)
                print(classifier.W.get_value())
                print("learning rate ", f(1))
                np.savetxt("parameters4", classifier.W.get_value(borrow=True))
            j = randint(0, item_vecs.shape[0] - 1)
            if randint(0,i2/batch_size) > 0:
                j = GetBestItem(item_vecs, item_bias, user_estimation[i], user_bias_estim[i], classifier.W.get_value(), users_used_items[i])#randint(0, item_vecs.shape[0] - 1)
            user = np.copy(user_estimation[i])
            user_b = np.copy(user_bias_estim[i])
            item_b = item_bias[j]
            item = item_vecs[j]
            reward1 = recieveAnswer(user_vecs[i], user_bias[i], item, item_b, global_bias, sigma)
            sucsess = 0
            if (reward1 > global_bias):
                users_used_items[i].add(j)
                sucsess = 1
            sample = make_input(item, item_b, user, user_b)
            user_estimation[i], user_bias_estim[i] = \
                OneStep(user_estimation[i], user_bias_estim[i], item, item_b, global_bias, reward1, learning_rate,
                learning_rate, learning_rate)
            #print(np.dot((user_estimation[i] - user), (user_estimation[i] - user).T))
            y1_arg = GetBestItem(item_vecs, item_bias, user_estimation[i], user_bias_estim[i], classifier.W.get_value(),
                    users_used_items[i])
            sample1 = make_input(item_vecs[y1_arg], item_bias[y1_arg], user_estimation[i], user_bias_estim[i])
            pred, new_prediction, err = train(sample, sample1, sucsess)#np.double(1. / (1. + math.exp(-reward))))
            #n_questions[i] += 1
            if (i2 % batch_size == 0):
                print(i2)
                print("Pred, err, reward, after", pred, err, sucsess, new_prediction)
        print("learning rate ", f(1))
        i1 += 1
#            print("suggested item ", y1_arg)
            #print ("user vector " , user_estimation[i])
            #print("item vector ", item)


    np.savetxt("parameters4", classifier.W.get_value(borrow=True))




def GenerateEx():
    truth = np.array([1., 2., 3.])
    x = np.random.rand(1000, 3)
    np.savetxt("x", x)
    y = np.dot(x, truth)
    np.savetxt("y", y)


if __name__ == '__main__':
    #DeleteMostPopularItems(100)
    #item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    #PrintItemPopularity(user_vecs, user_bias, item_vecs, item_bias, global_bias, 0)
    Learn()
    #Get_classifiers()
