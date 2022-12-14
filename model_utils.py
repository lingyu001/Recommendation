from tensorflow.python.keras.layers import Layer, Dropout, Activation, Lambda
from tensorflow.python.ops.init_ops_v2 import glorot_normal
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras import backend as K
import tensorflow as tf
class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [Activation(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = Activation(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class EmbeddingIndex(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NoMask(Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None

class PoolingLayer(Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        # a = concat_func(expand_seq_value_len_list)
        a = expand_seq_value_len_list[0] # temp change to only one
        if self.mode == "mean":
            hist = reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def l2_normalize(x, axis=-1):
    return Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)

# class SampledSoftmaxLayer(Layer):
#     def __init__(self, sampler_config, temperature=1.0, **kwargs):
#         self.sampler_config = sampler_config
#         self.temperature = temperature
#         self.sampler = self.sampler_config['sampler']
#         self.item_count = self.sampler_config['item_count']

#         super(SampledSoftmaxLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.vocabulary_size = input_shape[0][0]
#         self.zero_bias = self.add_weight(shape=[self.vocabulary_size],
#                                          initializer=Zeros,
#                                          dtype=tf.float32,
#                                          trainable=False,
#                                          name="bias")
#         super(SampledSoftmaxLayer, self).build(input_shape)

#     def call(self, inputs_with_item_idx, training=None, **kwargs):
#         item_embeddings, user_vec, item_idx = inputs_with_item_idx
#         if item_idx.dtype != tf.int64:
#             item_idx = tf.cast(item_idx, tf.int64)
#         user_vec /= self.temperature
#         if self.sampler == "inbatch":
#             item_vec = tf.gather(item_embeddings, tf.squeeze(item_idx, axis=1))
#             logits = tf.matmul(user_vec, item_vec, transpose_b=True)
#             loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)

#         else:
#             num_sampled = self.sampler_config['num_sampled']
#             if self.sampler == "frequency":
#                 sampled_values = tf.nn.fixed_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
#                                                                        self.vocabulary_size,
#                                                                        distortion=self.sampler_config['distortion'],
#                                                                        unigrams=np.maximum(self.item_count, 1).tolist(),
#                                                                        seed=None,
#                                                                        name=None)
#             elif self.sampler == "adaptive":
#                 sampled_values = tf.nn.learned_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
#                                                                          self.vocabulary_size, seed=None, name=None)
#             elif self.sampler == "uniform":
#                 try:
#                     sampled_values = tf.nn.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
#                                                                      self.vocabulary_size, seed=None, name=None)
#                 except AttributeError:
#                     sampled_values = tf.random.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
#                                                                          self.vocabulary_size, seed=None, name=None)
#             else:
#                 raise ValueError(' `%s` sampler is not supported ' % self.sampler)

#             loss = tf.nn.sampled_softmax_loss(weights=item_embeddings,
#                                               biases=self.zero_bias,
#                                               labels=item_idx,
#                                               inputs=user_vec,
#                                               num_sampled=num_sampled,
#                                               num_classes=self.vocabulary_size,
#                                               sampled_values=sampled_values
#                                               )
#         return tf.expand_dims(loss, axis=1)

#     def compute_output_shape(self, input_shape):
#         return (None, 1)

#     def get_config(self, ):
#         config = {'sampler_config': self.sampler_config, 'temperature': self.temperature}
#         base_config = super(SampledSoftmaxLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

class SampledSoftmaxLayer(Layer):
    def __init__(self, temperature=1.0, **kwargs):
        # self.sampler_config = sampler_config
        self.temperature = temperature # temperature = 1.0 so no impact
        # self.sampler = self.sampler_config['sampler']
        # self.item_count = self.sampler_config['item_count']

        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.vocabulary_size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.vocabulary_size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_item_idx, training=None, **kwargs):
        item_embeddings, user_vec, item_idx = inputs_with_item_idx
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)
        user_vec /= self.temperature
        # if self.sampler == "inbatch":
        #     item_vec = tf.gather(item_embeddings, tf.squeeze(item_idx, axis=1))
        #     logits = tf.matmul(user_vec, item_vec, transpose_b=True)
        #     loss = inbatch_softmax_cross_entropy_with_logits(logits, self.item_count, item_idx)

        # else:
        #     num_sampled = self.sampler_config['num_sampled']
        #     if self.sampler == "frequency":
        #         sampled_values = tf.nn.fixed_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
        #                                                                self.vocabulary_size,
        #                                                                distortion=self.sampler_config['distortion'],
        #                                                                unigrams=np.maximum(self.item_count, 1).tolist(),
        #                                                                seed=None,
        #                                                                name=None)
        #     elif self.sampler == "adaptive":
        #         sampled_values = tf.nn.learned_unigram_candidate_sampler(item_idx, 1, num_sampled, True,
        #                                                                  self.vocabulary_size, seed=None, name=None)
        #     elif self.sampler == "uniform":
        #         try:
        #             sampled_values = tf.nn.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
        #                                                              self.vocabulary_size, seed=None, name=None)
        #         except AttributeError:
        #             sampled_values = tf.random.uniform_candidate_sampler(item_idx, 1, num_sampled, True,
        #                                                                  self.vocabulary_size, seed=None, name=None)
        #     else:
        #         raise ValueError(' `%s` sampler is not supported ' % self.sampler)

        loss = tf.nn.sampled_softmax_loss(weights=item_embeddings,
                                              biases=self.zero_bias,
                                              labels=item_idx,
                                              inputs=user_vec,
                                              num_sampled=255,
                                              num_classes=self.vocabulary_size,
                                              sampled_values=None
                                              )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'sampler_config': self.sampler_config, 'temperature': self.temperature} 
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)

def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)