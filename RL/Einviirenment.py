from dataUtils import *

class Envierment(object):
    def __init__( self, expand, n_users, sigma, learning_rate):
        self.item_vecs, self.item_bias, self.user_vecs, self.user_bias, self.global_bias = GetData("data")
        self.latent_dim = self.user_vecs.shape[1]
        self.expand = expand
        self.sigma = sigma
        self.n_users = n_users
        self.learning_rate = learning_rate
        if (expand > 0):
            self.user_vecs, self.user_bias = \
                ExpandData(self.user_vecs, self.user_bias, expand)
        self.user_vecs = self.user_vecs[:n_users]
        self.user_bias = self.user_bias[:n_users]
        self.user_vecs_estim = np.zeros([n_users, self.latent_dim])
        self.user_bias_estim = np.zeros(n_users)
        self.actions = np.c_[self.item_vecs, self.item_bias]

    def reward(self, user_n, item_n):
        r = recieveAnswer( self.user_vecs[user_n], self.user_bias[user_n],
                              self.item_vecs[item_n], self.item_bias[item_n], self.global_bias, self.sigma)
        if (r > SUCSESS()):
            return 1.
        return -1.

    def update_state(self, user_n, item_n):
        self.user_vecs_estim[user_n], self.user_bias_estim[user_n] = \
            OneStep(self.user_vecs_estim[user_n], self.user_bias_estim[user_n],
                    self.item_vecs[item_n], self.item_bias[item_n], self.global_bias,
                    self.reward(user_n, item_n),
                    self.learning_rate,
                    self.learning_rate, self.learning_rate)