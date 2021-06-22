import tensorflow as tf
from tensorflow.keras import layers
import copy

class SequentialLayer(tf.keras.layers.Layer):
    def __init__(self , args):
        super().__init__()
        self.not_base = True
        self.wrap_list = []
        for layer in args:self.wrap_list.append(copy.deepcopy(layer))
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        for l in self.wrap_list:
            inputs = l(inputs,training=training)
        return inputs

    def call_funtion(self,x):
        for l in self.wrap_list:
            if hasattr(l,'not_base') and l.not_base:
                x = l.call_funtion(x)
            else:
                x  = l(x)
        return x