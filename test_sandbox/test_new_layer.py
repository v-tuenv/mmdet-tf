import tensorflow as tf


class L(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        self.add_loss({'all_s':tf.math.reduce_sum(x)})
        self.add_metric(tf.math.reduce_sum(x), name='all_s')
        return x


model = tf.keras.Sequential([L()])
model.build((None,6))
data = tf.random.normal(shape=(100,6))
tf.print(tf.math.reduce_sum(data))
model.compile()
model.fit(data,epochs=1, batch_size=100)
print(model.losses, model.layers[0].losses)