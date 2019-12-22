# +
'''
Graph convolutional layer for tf.keras.
A tf.keras version of https://github.com/tkipf/keras-gcn/blob/master/kegra/layers/graph.py
'''

import tensorflow as tf
try:
    from tensorflow.keras import activations, initializers, constraints, regularizers, layers
    from tensorflow.keras import backend as K
except ModuleNotFoundError:
    from tensorflow.python.keras import activations, initializers, constraints, regularizers, layers
    from tensorflow.python.keras import backend as K

class GINonStack(layers.Layer):
    '''Basic graph convolution layer as in https://arxiv.org/abs/1609.02907'''
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='ones',
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GINonStack, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # Not sure this holds
        # self.supports_masking = True

    def compute_output_shape(self, input_shape):
        ''' Necessary to calculate shape transformation of layer

        input_shape will be (batch_size, nodes, nodes, 2*features)
        '''
        # print(input_shape)
        output_shape = (input_shape[0:2], self.units)
        return output_shape  # (batch_size, nodes, units)

    def build(self, input_shape):
        ''' Construct the layer '''
        assert len(input_shape) == 4
        n_nodes = int(input_shape.dims[1])
        n_2features = int(input_shape.dims[3])

        # Weight matrix W in original paper
        self.kernel = self.add_weight(
            shape=(n_2features, self.units),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        # Optional biases to add (not shown in original paper)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        # Must set at end of build function
        super(GINonStack, self).build(input_shape)
        self.built = True

    def call(self, inputs):
        ''' Operation called during processing

        inputs will be (batch_size, nodes, nodes, 2*features)
        '''
        
        # First sum across the row (or is it columns? doesn't matter)
        # This could also be a reduce_mean?
        # (b,n,n,2f) -> (b,n,2f)
        output = tf.reduce_sum(inputs, axis=2)
        
        # Multiply by weights
        # (b,n,2f) * (2f,u) = (b,n,u)
        output = K.dot(output, self.kernel)

        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': initializers.serialize(
                self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(
                self.bias_regularizer),
            'kernel_constraint': constraints.serialize(
                self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GINonStack, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
