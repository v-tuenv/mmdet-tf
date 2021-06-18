import tensorflow as tf
from tensorflow.keras import layers
import copy

class SequentialLayer(tf.keras.layers.Layer):
    def __init__(self , args):
        super().__init__()
        self.wrap_list = []
        for layer in args:self.wrap_list.append(copy.deepcopy(layer))
    def call(self, inputs):
        for l in self.wrap_list:
            inputs = l(inputs)
        return inputs