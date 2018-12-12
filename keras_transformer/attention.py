import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects


class _BaseMultiHeadAttention(Layer):
    """
    Base class for two types of Multi-head attention layers:
    Self-attention and its more general form used in decoders (the one which
    takes values and keys from the encoder).
    """
    def __init__(self, num_heads: int, use_masking: bool,
                 dropout: float = 0.0,
                 compression_window_size: int = None,
                 **kwargs):
        """
        :param num_heads: number of attention heads
        :param use_masking: when True, forbids the attention to see the further
          elements in the sequence (particularly important in language
          modelling).
        :param dropout: dropout that should be applied to the attention
          (after the softmax).
        :param compression_window_size: an integer value >= 1 controlling
          how much we should compress the attention. For more details,
          read about memory-compressed self-attention in
          "Generating Wikipedia by summarizing long sequences"
          (https://arxiv.org/pdf/1801.10198.pdf).
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.dropout = dropout
        if (compression_window_size is not None
                and compression_window_size <= 0):
            assert ValueError(
                f"Too small compression window ({compression_window_size})")
        self.compression_window_size = compression_window_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['use_masking'] = self.use_masking
        config['dropout'] = self.dropout
        config['compression_window_size'] = self.compression_window_size
        return config

    # noinspection PyAttributeOutsideInit
    def build_output_params(self, d_model):
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(d_model, d_model),
            initializer='glorot_uniform',
            trainable=True)
        if self.compression_window_size is not None:
            self.k_conv_kernel = self.add_weight(
                name='k_conv_kernel',
                shape=(self.compression_window_size,
                       d_model // self.num_heads,
                       d_model // self.num_heads),
                initializer='glorot_uniform',
                trainable=True)
            self.k_conv_bias = self.add_weight(
                name='k_conv_bias',
                shape=(d_model // self.num_heads,),
                initializer='zeros',
                trainable=True)
            self.v_conv_kernel = self.add_weight(
                name='v_conv_kernel',
                shape=(self.compression_window_size,
                       d_model // self.num_heads,
                       d_model // self.num_heads),
                initializer='glorot_uniform',
                trainable=True)
            self.v_conv_bias = self.add_weight(
                name='v_conv_bias',
                shape=(d_model // self.num_heads,),
                initializer='zeros',
                trainable=True)

    def validate_model_dimensionality(self, d_model: int):
        if d_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')

    def attention(self, pre_q, pre_v, pre_k, out_seq_len: int, d_model: int,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param out_seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        if self.compression_window_size is None:
            k_transposed = K.permute_dimensions(pre_k, [0, 2, 3, 1])
        else:
            # Memory-compressed attention described in paper
            # "Generating Wikipedia by Summarizing Long Sequences"
            # (https://arxiv.org/pdf/1801.10198.pdf)
            # It compresses keys and values using 1D-convolution which reduces
            # the size of Q * K_transposed from roughly seq_len^2
            # to convoluted_seq_len^2. If we use strided convolution with
            # window size = 3 and stride = 3, memory requirements of such
            # memory-compressed attention will be 9 times smaller than
            # that of the original version.
            if self.use_masking:
                raise NotImplementedError(
                    "Masked memory-compressed attention has not "
                    "been implemented yet")
            k = K.permute_dimensions(pre_k, [0, 2, 1, 3])
            k, v = [
                K.reshape(
                    # Step 3: Return the result to its original dimensions
                    # (batch_size, num_heads, seq_len, d_model//heads)
                    K.bias_add(
                        # Step 3: ... and add bias
                        K.conv1d(
                            # Step 2: we "compress" K and V using strided conv
                            K.reshape(
                                # Step 1: we reshape K and V to
                                # (batch + num_heads,  seq_len, d_model//heads)
                                item,
                                (-1,
                                 K.int_shape(item)[-2],
                                 d_model // self.num_heads)),
                            kernel,
                            strides=self.compression_window_size,
                            padding='valid', data_format='channels_last'),
                        bias,
                        data_format='channels_last'),
                    # new shape
                    K.concatenate([
                        K.shape(item)[:2],
                        [-1, d_model // self.num_heads]]))
                for item, kernel, bias in (
                    (k, self.k_conv_kernel, self.k_conv_bias),
                    (v, self.v_conv_kernel, self.v_conv_bias))]
            k_transposed = K.permute_dimensions(k, [0, 1, 3, 2])
        # shaping K into (batch_size, num_heads, d_model//heads, seq_len)
        # for further matrix multiplication
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads),
                            dtype=K.floatx())
        q_shape = K.int_shape(q)
        k_t_shape = K.int_shape(k_transposed)
        v_shape = K.int_shape(v)
        # before performing batch_dot all tensors are being converted to 3D
        # shape (batch_size * num_heads, rows, cols) to make sure batch_dot
        # performs identically on all backends
        attention_heads = K.reshape(
            K.batch_dot(
                self.apply_dropout_if_needed(
                    K.softmax(
                        self.mask_attention_if_needed(
                            K.batch_dot(
                                K.reshape(q, (-1,) + q_shape[-2:]),
                                K.reshape(k_transposed,
                                          (-1,) + k_t_shape[-2:]))
                            / sqrt_d)),
                    training=training),
                K.reshape(v, (-1,) + v_shape[-2:])),
            (-1, self.num_heads, q_shape[-2], v_shape[-1]))
        attention_heads_merged = K.reshape(
            K.permute_dimensions(attention_heads, [0, 2, 1, 3]),
            (-1, d_model))
        attention_out = K.reshape(
            K.dot(attention_heads_merged, self.output_weights),
            (-1, out_seq_len, d_model))
        return attention_out

    def apply_dropout_if_needed(self, attention_softmax, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(attention_softmax, self.dropout)

            return K.in_train_phase(dropped_softmax, attention_softmax,
                                    training=training)
        return attention_softmax

    def mask_attention_if_needed(self, dot_product):
        """
        Makes sure that (when enabled) each position
        (of a decoder's self-attention) cannot attend to subsequent positions.
        This is achieved by assigning -inf (or some large negative number)
        to all invalid connections. Later softmax will turn them into zeros.
        We need this to guarantee that decoder's predictions are based
        on what has happened before the position, not after.
        The method does nothing if masking is turned off.
        :param dot_product: scaled dot-product of Q and K after reshaping them
        to 3D tensors (batch * num_heads, rows, cols)
        """
        if not self.use_masking:
            return dot_product
        last_dims = K.int_shape(dot_product)[-2:]
        low_triangle_ones = (
            np.tril(np.ones(last_dims))
            # to ensure proper broadcasting
            .reshape((1,) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (
            K.constant(low_triangle_ones, dtype=K.floatx()) * dot_product +
            K.constant(close_to_negative_inf * inverse_low_triangle))
        return result


class MultiHeadAttention(_BaseMultiHeadAttention):
    """
    Multi-head attention which can use two inputs:
    First: from the encoder - it's used to project the keys and the values
    Second: from the decoder - used to project the queries.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise ValueError(
                'You must call this layer passing a list of two tensors'
                '(for keys/values and queries)')
        values_dim, query_dim = input_shape[0][-1], input_shape[1][-1]
        if query_dim != values_dim:
            raise ValueError(
                f'Both keys/value and query inputs must be '
                f'of the same dimensionality, instead of '
                f'{values_dim} and {query_dim}.')
        d_model = query_dim
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_k and W_v which
        # are, in turn, concatenated W matrices of keys, and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.kv_weights = self.add_weight(
            name='kv_weights', shape=(d_model, d_model * 2),
            initializer='glorot_uniform', trainable=True)
        self.q_weights = self.add_weight(
            name='q_weights', shape=(d_model, d_model),
            initializer='glorot_uniform', trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not (isinstance(inputs, list) and len(inputs) == 2):
            raise ValueError(
                'You can call this layer only with a list of two tensors '
                '(for keys/values and queries)')
        key_values_input, query_input = inputs
        _, value_seq_len, d_model = K.int_shape(key_values_input)
        query_seq_len = K.int_shape(inputs[1])[-2]
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        kv = K.dot(K.reshape(key_values_input, [-1, d_model]), self.kv_weights)
        # splitting the keys, the values and the queries before further
        # processing
        pre_k, pre_v = [
            K.reshape(
                # K.slice(kv, (0, i * d_model), (-1, d_model)),
                kv[:, i * d_model: (i + 1) * d_model],
                (-1, value_seq_len,
                 self.num_heads, d_model // self.num_heads))
            for i in range(2)]
        pre_q = K.reshape(
            K.dot(K.reshape(query_input, [-1, d_model]), self.q_weights),
            (-1, query_seq_len, self.num_heads, d_model // self.num_heads))
        return self.attention(pre_q, pre_v, pre_k, query_seq_len, d_model,
                              training=kwargs.get('training'))


class MultiHeadSelfAttention(_BaseMultiHeadAttention):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input and has implementation which is better suited for
    such use case that more general MultiHeadAttention class.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')
        d_model = input_shape[-1]
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_q, W_k and W_v which
        # are, in turn, concatenated W matrices of keys, queries and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_uniform',
            trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = K.int_shape(inputs)
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        qkv = K.dot(K.reshape(inputs, [-1, d_model]), self.qkv_weights)
        # splitting the keys, the values and the queries before further
        # processing
        pre_q, pre_k, pre_v = [
            K.reshape(
                # K.slice(qkv, (0, i * d_model), (-1, d_model)),
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model,
                                       training=kwargs.get('training'))
        return attention_out

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'MultiHeadAttention': MultiHeadAttention,
})
