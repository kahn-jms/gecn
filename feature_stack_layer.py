# +
'''
Create a populated stack of the adjacency matrix with node features filled in
'''

import tensorflow as tf
try:
    from tensorflow.keras import activations, initializers, constraints, regularizers, layers
    from tensorflow.keras import backend as K
except ModuleNotFoundError:
    from tensorflow.python.keras import activations, initializers, constraints, regularizers, layers
    from tensorflow.python.keras import backend as K

class FeatureStack(layers.Layer):
    '''Embed connected node pairs and apply a graph convolution to them'''
    def __init__(
        self,
        **kwargs
    ):
        super(FeatureStack, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        ''' Necessary to calculate shape transformation of layer

        input_shape will be (feature_matrices, laplacian_matrices)
        feature_matrices = (batch_size, nodes, features)
        laplacian_matrices = (bathc_size, nodes, nodes)
        '''
        # print(input_shape)
        features_shape = input_shape[0]
        output_shape = (features_shape[0:2], feature_shape[1], (2 * feature_shape[-1]))
        return output_shape  # (batch_size, nodes, output_dim)

    def build(self, input_shape):
        ''' Construct the layer '''
        features_shape = input_shape[0]
        assert len(features_shape) == 3
        n_nodes = int(features_shape.dims[1])
        n_features = int(features_shape.dims[2])

        # Must set at end of build function
        super(FeatureStack, self).build(input_shape)
        self.built = True

    def call(self, inputs):
        ''' Operation called during processing

        inputs will be (feature_matrices, laplacian_matrices)
        feature_matrices = (batch_size, nodes, features)
        laplacian_matrices = (batch_size, nodes, nodes)
        '''
        features = inputs[0]
        laplacian = inputs[1]

        # First need to populate the adjacency matrix with the feature values
        # For f features the we will have a matrix that is (b, n, n, 2f) where
        # the 2f are the features of the two nodes that that coordinate represents a
        # connection between
        # NOTE: For ease of visualising tf.eval() calls I'll do everything with channel-first
        # and switch to channel last at the end
        
        # First create the two adjacency stack
        # (b,f,n,n)
        adj_stack = tf.stack([laplacian] * features.shape[-1].value, axis=1)
        
        # Need to swap the feature axes for the element-wise multiply to work
        # (b,n,f) -> (b,f,n)
        t_feats = tf.transpose(features, perm=[0, 2, 1])
        
        # Element-wise multiply the adjacency matrix with the feature matrix along the rows and columns
        # (b,f,n,n) each
        row_stack = tf.multiply(adj_stack, tf.expand_dims(t_feats,-1))
        col_stack = tf.multiply(adj_stack, tf.expand_dims(t_feats,-2))
        
        # And stack the two to make our node pair feature adjacency matrix
        # (b,2f,n,n)
        feat_stack = tf.concat([row_stack, col_stack], axis=1)
        
        # And make channel-last as promised
        # (b,n,n,2f)
        feat_stack = tf.transpose(feat_stack, perm=[0, 2, 3, 1])

        return feat_stack

    def get_config(self):
        config = {
        }

        base_config = super(FeatureStack, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
