import tensorflow as tf


class PCAProjection(tf.keras.layers.Layer):
    def build(self, input_shape):
        n = input_shape[-1]
        self.mean = self.add_weight(
            name="mean",
            shape=(1, n),
            initializer="zeros",
            trainable=False,
        )
        self.pca = self.add_weight(
            name="pca",
            shape=(n, n),
            initializer="zeros",
            trainable=False,
        )

    def call(self, x):
        return (x - self.mean) @ self.pca