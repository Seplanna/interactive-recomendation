import theano as th
import numpy as np
import theano.tensor as T
from random import gauss
from random import randint
import math

class Easy(object):
    def __init__(self, n_in):
        self.W = th.shared(
            value=np.zeros(
                (n_in),
                dtype=th.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.params = [self.W]

    def cost(self, x, y):
        return (T.dot(x, self.W) - y) ** 2

def main2():
    x1 = np.genfromtxt("x")
    y1 = np.genfromtxt("y")
    c = Easy(3)

    x = T.vector("x")
    y = T.dscalar("y")
    #    w = th.shared(np.random.randn(3), name="w")

    cost = (T.dot(x, c.W) - y) ** 2
    g_W = T.grad(cost, c.W)
    learning_rate = 0.1
    updates = [c.W, c.W - learning_rate * g_W]

    prediction = np.dot(c.W, x)
    train = th.function(
        inputs=[x, y],
        outputs=[prediction, cost],
        updates=[updates])

    for i in range(x1.shape[0]):
        pred, err = train(x1[i], y1[i])
        print(c.W.get_value(borrow=True))

    save_file = open('model', 'wb')  # this will overwrite current contents
    # cPickle.dump(c.W.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
