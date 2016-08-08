import theano as th
import theano.tensor as T
from dataUtils import *
import numpy as np
from random import gauss
from random import randint
import math

def First_approximation(latent_dim):
    W = np.zeros(3 * latent_dim + 2)
    W[latent_dim] = 1.
    for i in range(latent_dim):
        W[-i - 1] = 1.
    return W

class LogisticRegression(object):

    def __init__(self, first_W):
        self.W = th.shared(
            value=first_W,
            name='W',
            borrow=True
        )
        self.params = [self.W]

    def cost(self, input):
        # s = T.sum(self.b) + T.dot(input, self.W)
        s = T.dot(input, self.W)
        # return  1. / (1 + T.exp(-s))
        return s

class Qlearning(object):

    def __init__(self, first_W):
        self.log_regression = LogisticRegression(first_W)
        st = T.dvector('st')
        ac = T.dvector('ac')
        self.q_ = th.function(inputs=[st, ac],
                              outputs=[self.log_regression.cost(T.concatenate([ac, st, ac[:-1] * st[:-1]]))])

    def q(self, state, action):
        return self.log_regression.cost(T.concatenate([action, state, action[:-1] * state[:-1]]))

    def recieve_new_greedy_action(self, actions, state, users_used_items):
        max_q = self.q_(state, actions[0])
        best_action = actions[0]
        action_numerator = 0
        first_estim = False
        for i,a in enumerate(actions):
            new_q = self.q_(state, a)
            if ((new_q > max_q or not first_estim)  and i not in users_used_items):
                max_q = new_q
                best_action = a
                action_numerator = i
                first_estim = True
        return [max_q, best_action,  action_numerator]

    def recieve_e_greedy_action(self, actions, state, n, users_used_items):
        if randint(0, n) > 0:
            return self.recieve_new_greedy_action(actions, state, users_used_items)
        j = randint(0, actions.shape[0] - 1)
        action = actions[j]
        return [self.q_(state, actions[j]), action, j]
