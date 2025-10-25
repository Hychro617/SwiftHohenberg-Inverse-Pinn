import tensorflow as tf
from keras.layers import Layer
from keras import backend as K
from keras.initializers import Constant

class RBFLayer(Layer):
    def __init__(self, units, beta, max_val, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.init_beta = K.cast_to_floatx(beta)
        self.max_val = max_val

    def build(self, input_shape):
        self.mu = self.add_weight(
            name="mu",
            shape=(int(input_shape[1]), self.units),
            initializer=tf.random_uniform_initializer(
                minval=0, maxval=self.max_val, seed=None
            ),
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.units,),
            initializer=Constant(value=self.init_beta),
            trainable=True,
        )
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.beta * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)