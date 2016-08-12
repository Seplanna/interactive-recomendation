#from IPython.display import Image, display
import wget
from PIL import Image
import numpy as np
import requests
import json
import math
import os.path
from multiprocessing import Process
from multiprocessing import Pool
from Einviirenment import *
from logistic_regression import*
import matplotlib.pyplot as plt

headers = {'Accept': 'application/json'}
payload = {'api_key': "e16fe7a4d1f7e73c8d9a611656c980c8"}
#response1 = requests.get("http://api.themoviedb.org/3/configuration", \
#                       params=payload, \
#                       headers=headers)
#response1 = json.loads(response1.text)
#base_url = response1['images']['base_url'] + 'w185'

#SUCSESS = 3.8

def GetItemsNames(file):
    items_names = {}
    with open(file) as names:
        for line in names:
            line = line.strip().split('|')
            items_names[int(line[0])] = [line[1], line[4]]
    return items_names

def get_poster(imdb_url, base_url, api_key = "e16fe7a4d1f7e73c8d9a611656c980c8"):
    # Get IMDB movie ID
    images = 'images/'
    response = requests.get(imdb_url)
    movie_id = response.url.split('/')[-2]
    
    # Query themoviedb.org API for movie poster path.
    movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(movie_id)
    headers = {'Accept': 'application/json'}
    payload = {'api_key': api_key} 
    response = requests.get(movie_url, params=payload, headers=headers)
    try:
        file_path = json.loads(response.text)['posters'][0]['file_path']
    except:
        # IMDB movie ID is sometimes no good. Need to get correct one.
        movie_title = imdb_url.split('?')[-1].split('(')[0]
        payload['query'] = movie_title
        response = requests.get('http://api.themoviedb.org/3/search/movie',\
                                params=payload,\
                                headers=headers)
        try:
            movie_id = json.loads(response.text)['results'][0]['id']
            payload.pop('query', None)
            movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'\
                        .format(movie_id)
            response = requests.get(movie_url, params=payload, headers=headers)
            file_path = json.loads(response.text)['posters'][0]['file_path']
        except:
            # Sometimes the url just doesn't work.
            # Return '' so that it does not mess up Image()
            return ''
    poster = base_url + file_path
    filename = wget.download(poster)
    im = Image.open(filename)
    im.show()
    os.remove(filename)
    #display(Image(poster))
    return poster

def OurApproachOneUser(classifier, env, user_n, n_q, distanse_to_real):
    answers = []
    users_used_items = set()
    for i in range(n_q):
        user = np.append(env.user_vecs_estim[user_n], env.user_bias_estim[user_n])
        max_q, best_item, item = classifier.recieve_new_greedy_action(env.actions, user, users_used_items)
        reward = env.reward(user_n, item)
        users_used_items.add(item)
        answers.append(reward)
        env.update_state(user_n, item)
        d = env.user_vecs_estim[user_n] - env.user_vecs[user_n]
        d = np.dot(d, d.T)
        distanse_to_real[i] += d
    #print(answers)
    return answers

def OurApproach(args):
    #items_names = GetItemsNames("data/u.item")
    #item_vecs, item_bias, user_estimation, user_bias_estim, global_bias1 = GetData("data1")
    W = args[0]
    file = args[1]
    n_thread = args[2]
    n_threads = args[3]

    n_q = 40
    dis_to_real = [0 for i in range(n_q)]
    expand = 1
    learning_rate = 0.001
    n_users = 40000
    sigma = 0.5
    env = Envierment(expand, n_users, sigma, learning_rate)
    classifier = Qlearning(first_W=W)
    first_user = (env.n_users / n_threads) * n_thread
    with open(file, 'w') as result_file:
        for u in range(first_user, first_user + env.n_users / n_threads):
            #if (u == 100):
            #    break
            if(u % 2 == 0):
                print(u)
            answers = OurApproachOneUser(classifier, env, u, n_q, dis_to_real)
            if -1 in answers:
                print(answers)
            #print(dis_to_real)
            result_file.write('\t'.join(str(a) for a in answers))
            result_file.write('\n')
    a = range(n_q)
    plt.plot(a, np.array(dis_to_real) / n_users, 'r')
    plt.show()

def Play():
    W = np.genfromtxt("parameters")
    items_names= GetItemsNames("data/u.item")
    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")


    y1_arg = 0
    element = -1000
    print("LET's PLAY")
    print("Play until you write finish")
    s = '0'

    users_used_items = set()
    user_estimation = np.zeros(user_vecs.shape[1])
    user_bias_estim = np.zeros(1)
    s1 = '0'
    n_q = 0
    while not s1 == 'finish':
        element = -1000
        learning_rate = 1./math.sqrt(n_q + 1.)
        #print ('user estimation = ', user_estimation)
        for item1 in range(0, item_vecs.shape[0]):
            c = np.dot(W, make_input(item_vecs[item1], item_bias[item1],
                                                        user_estimation, user_bias_estim))
            if (element < c and not (item1 in users_used_items)):
                element = c
                y1_arg = item1
            item = items_names[y1_arg + 1]

        get_poster(item[1], base_url)
        print("user estimation ", user_estimation)
        print(item[0], element, np.dot(user_estimation, item_vecs[y1_arg]), user_bias_estim, item_bias[y1_arg])
        s = raw_input("how do you like item " + str(y1_arg) + "\n")
        OneStep(user_estimation, user_bias_estim, item_vecs[y1_arg], item_bias[y1_arg],
                global_bias, float(s), learning_rate,
                learning_rate, learning_rate)
        users_used_items.add(y1_arg)
        print(users_used_items)
        n_q += 1

if __name__ == '__main__':
    #GreedyApproach()
    #Play()
    W = np.genfromtxt("parameters")
    latent_dim = 10
    W1 = np.zeros(W.shape[0])
    W1[latent_dim] = 1.
    for i in range(latent_dim):
        W1[-i - 1] = 1.

    n_th = 4
    args_g = []
    for i in range(n_th):
        args_g.append([W1, "GreedyPlay" + "_" + str(i), i, n_th])
    #OurApproach(W1, "GreedyPlay")
    args = []
    for i in range(n_th):
        args.append([W, "OurApproach3" + "_" + str(i), i, n_th])

    #p = Pool(processes=n_th)
    #print(args)
    #p.map(OurApproach, args)
    #p1 = Pool(processes=n_th)
    #p1.map(OurApproach, args_g)

    OurApproach([W1, "GreedyPlay", 0, 1])
    #OurApproach([W, "OurApproach3", 0, 1])