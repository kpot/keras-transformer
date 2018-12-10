"""
Tools that are not necessary for the Transformer by itself, but might be
useful in building models with it.
"""
import math

from keras import activations, regularizers
# noinspection PyPep8Naming
from keras import backend as K
from keras.engine import Layer
from keras.layers import Embedding
from keras.utils import get_custom_objects


class ReusableEmbedding(Embedding):
    """
    A "reusable" form of the Embedding layer, which returns its
    full embedding matrix as one of the outputs.
    This is necessary to guarantee correct work of Keras when the matrix
    is being re-used again in TiedOutputEmbedding layer.
    """
    def call(self, inputs, **kwargs):
        result = super().call(inputs, **kwargs)
        return [result, self.embeddings]

    def compute_output_shape(self, input_shape):
        return [super().compute_output_shape(input_shape),
                K.int_shape(self.embeddings)]

    def compute_mask(self, inputs, mask=None):
        return [super().compute_mask(inputs, mask), None]


class TiedOutputEmbedding(Layer):
    """
    Allows to reuse the same word embedding matrix both for the input and
    the output layers of the network.
    This is called Weight Tying and is proven to improve performance
    of neural network language models, as well as decrease their number
    of parameters (eliminating the need for a separate huge matrix
    of output weights).

    The layers is supposed to be called with two inputs, like

        TiedOutputEmbedding()([main_input, embedding_matrix])

    where the `main_input` is the output of the previous layer (like LSTM)
    and the `embedding_matrix` coming from the `ReusableEmbedding` layer.

    https://arxiv.org/abs/1608.05859
    https://arxiv.org/abs/1611.01462
    https://blog.openai.com/language-unsupervised/
    """
    def __init__(self, activation=None,
                 add_biases=False, projection_regularizer=None,
                 projection_dropout: float = 0.0,
                 scaled_attention=False,
                 **kwargs):
        self.activation = activations.get(activation)
        self.add_biases = add_biases
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.projection_dropout = projection_dropout
        self.scaled_attention = scaled_attention
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return dict(
            config,
            activation=activations.serialize(self.activation),
            add_biases=self.add_biases,
            projection_regularizer=regularizers.serialize(
                self.projection_regularizer),
            projection_dropout=self.projection_dropout,
            scaled_attention=self.scaled_attention)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        main_input_shape, embedding_matrix_shape = input_shape
        emb_input_dim, emb_output_dim = embedding_matrix_shape
        assert len(main_input_shape) == 3
        self.projection = self.add_weight(
            name='kernel',
            shape=(main_input_shape[-1], emb_output_dim),
            initializer='glorot_uniform',
            regularizer=self.projection_regularizer,
            trainable=True)
        if self.add_biases:
            self.biases = self.add_weight(
                name='biases',
                shape=(emb_output_dim,),
                initializer='zeros',
                trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        main_input, embedding_matrix = inputs
        input_shape_tensor = K.shape(main_input)
        last_input_dim = K.int_shape(main_input)[-1]
        emb_input_dim, emb_output_dim = K.int_shape(embedding_matrix)
        projected = K.dot(K.reshape(main_input, (-1, last_input_dim)),
                          self.projection)
        if self.add_biases:
            projected = K.bias_add(projected, self.biases,
                                   data_format='channels_last')
        if 0 < self.projection_dropout < 1:
            projected = K.in_train_phase(
                lambda: K.dropout(projected, self.projection_dropout),
                projected,
                training=kwargs.get('training'))
        attention = K.dot(projected, K.transpose(embedding_matrix))
        if self.scaled_attention:
            # scaled dot-product attention, described in
            # "Attention is all you need" (https://arxiv.org/abs/1706.03762)
            sqrt_d = K.constant(math.sqrt(emb_output_dim), dtype=K.floatx())
            attention = attention / sqrt_d
        result = K.reshape(
            self.activation(attention),
            (input_shape_tensor[0],
             input_shape_tensor[1],
             emb_input_dim))
        return result

    def compute_output_shape(self, input_shape):
        main_input_shape, embedding_matrix_shape = input_shape
        emb_input_dim, emb_output_dim = embedding_matrix_shape
        return main_input_shape[0], main_input_shape[1], emb_input_dim


get_custom_objects().update({
    'ReusableEmbedding': ReusableEmbedding,
    'TiedOutputEmbedding': TiedOutputEmbedding,
})
