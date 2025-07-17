import tensorflow as tf


class SquareLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def call(self, inputs: tf.Tensor):
        return tf.concat([inputs, inputs * inputs], -1)
