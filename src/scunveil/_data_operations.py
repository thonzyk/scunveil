import tensorflow as tf
import numpy as np


def logits_to_CPM(pred, add_oder_of_magnitude=6.0):
    pred = tf.cast(tf.convert_to_tensor(pred), tf.float32)
    log10_pred = tf.nn.log_softmax(pred, axis=-1) / tf.math.log(10.0)
    pred = log10_pred + add_oder_of_magnitude

    pred = pred.numpy()

    return pred


def simple_scipy_norm_x(x):
    x = x.tocoo()
    idx = np.stack([x.row, x.col], axis=1).astype(np.int64)
    shape = np.array(x.shape, dtype=np.int64)

    vals = tf.constant(x.data.astype(np.float32))
    vals = tf.math.log1p(vals)

    sp = tf.sparse.SparseTensor(idx, vals, shape)
    sp = tf.sparse.reorder(sp)

    x_tf_dense = tf.sparse.to_dense(sp)
    return x_tf_dense
