#from IPython.display import Image, display
import wget
from PIL import Image
import numpy as np
import requests
import json
import math
import os.path
from dataUtils import *


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

def GreedyForOneUser(user, user_bias, item_vecs, item_bias, global_bias, n_q):
    user_estim = np.zeros(item_vecs.shape[1])
    user_bias_estim = 0
    answers = []
    first_item = 1448
    user_used_items = []
    s = recieveAnswer(user, user_bias, item_vecs[first_item], item_bias[first_item], global_bias, 0.01)
    answers.append(int(s>SUCSESS))
    item = first_item
    for i in range(n_q):
        learning_rate = 1. / math.sqrt(i + 1.)
        OneStep(user_estim, user_bias_estim, item_vecs[item], item_bias[item],
                global_bias, float(s), learning_rate,
                learning_rate, learning_rate)
        element = -1000
        for item1 in range(0, item_vecs.shape[0]):
            c = np.dot(user_estim, item_vecs[item1]) + item_bias[item1] + user_bias_estim
            if (element < c and not (item1 in user_used_items)):
                element = c
                item = item1
        user_used_items.append(item)
        s = recieveAnswer(user, user_bias, item_vecs[item], item_bias[item], global_bias, 0.01)
        answers.append(int( s > SUCSESS))
    return answers

def GreedyApproach():
    items_names= GetItemsNames("data/u.item")
    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    n_users = user_vecs.shape[0]
    users_answers = []
    with open("GreedyPlay", 'w') as result_file:
        for u in range(n_users):
            answers = GreedyForOneUser(user_vecs[u], user_bias[u], item_vecs, item_bias, global_bias, 30)
            result_file.write('\t'.join(str(a) for a in answers))
            result_file.write('\n')

def OurApproachOneUser(user, user_bias, item_vecs, item_bias, global_bias, W, n_q, item_names,
                       user_estim, user_bias_estim):
    item = 0
    user_used_items = []
    answers = []

    for i in range(n_q):
        element = -1000
        learning_rate = 0.001
        #learning_rate = 1. / math.sqrt(i + 1.)
        for item1 in range(0, item_vecs.shape[0]):
            c = np.dot(W, make_input(item_vecs[item1], item_bias[item1],
                                 user_estim, user_bias_estim))
            if ((element < c or element == -1000) and not (item1 in user_used_items)):
                element = c
                item = item1
        #get_poster(item_names[item+1][1], base_url)
        #print(item_names[item+1][0])
        user_used_items.append(item)
        s = recieveAnswer(user, user_bias, item_vecs[item], item_bias[item], global_bias, 0.5)
        answers.append(int(s > SUCSESS()))
        sucsess = -1
        if (s > SUCSESS()):
            sucsess = 1
        OneStep(user_estim, user_bias_estim, item_vecs[item], item_bias[item],
                global_bias, sucsess, learning_rate,
                learning_rate, learning_rate)
        #print(np.dot((user_estim - user), (user_estim - user).T))
    print(answers)
    return answers

def OurApproach(W, file):
    items_names = GetItemsNames("data/u.item")
    item_vecs, item_bias, user_vecs, user_bias, global_bias = GetData("data")
    user_estimation = np.zeros((user_vecs.shape[0], user_vecs.shape[1]))
    user_bias_estim = np.zeros(user_vecs.shape[0])
    #item_vecs, item_bias, user_estimation, user_bias_estim, global_bias1 = GetData("data1")

    n_users = user_vecs.shape[0]
    users_answers = []
    with open(file, 'w') as result_file:
        for u in range(n_users):
            if (u == 100):
                break
            if(u % 2 == 0):
                print(u)
            answers = OurApproachOneUser(user_vecs[u], user_bias[u], item_vecs, item_bias, global_bias, W, 20, items_names,
                                         user_estimation[u], user_bias_estim[u])
            result_file.write('\t'.join(str(a) for a in answers))
            result_file.write('\n')

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
    W = np.genfromtxt("parameters4")
    latent_dim = 10
    W1 = np.zeros(W.shape[0])
    W1[latent_dim] = 1.
    for i in range(latent_dim):
        W1[-i - 1] = 1.
    OurApproach(W1, "GreedyPlay")
    OurApproach(W, "OurApproach3")