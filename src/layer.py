import numpy as np
import tensorflow as tf

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.leaky_relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_weight'):
            self.weight = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):        
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.weight)
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            x = tf.contrib.layers.bias_add(x)
            outputs = self.act(x)
        return outputs


class GCN():
    def __init__(self, adj, feature_dim, hidden_dim, name, dropout=0.):
        self.name = name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.adj = adj
        self.dropout = dropout
        
    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            embedding = GraphConvolution(
                name='layer_1',
                input_dim=self.feature_dim,
                output_dim=self.hidden_dim,
                adj=self.adj,
                act=tf.nn.leaky_relu,
                dropout=self.dropout)(inputs)
        return embedding

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)