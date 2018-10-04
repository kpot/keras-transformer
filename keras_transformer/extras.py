"""
Tools that are not necessary for the Transformer by itself, but might be
useful in building models with it.
"""

from keras import activations, regularizers, backend as K
from keras.engine import Layer
from keras.layers import Embedding


class TiedOutputEmbedding(Layer):
    """
    Allows to reuse the same word embedding matrix both for the input and
    the output layers of the network.
    This is called Weight Tying and is proven to improve performance
    of neural network language models, as well as decrease their number
    of parameters (eliminating the need for a separate huge matrix
    of output weights).

    https://arxiv.org/abs/1608.05859
    https://arxiv.org/abs/1611.01462
    https://blog.openai.com/language-unsupervised/
    """
    def __init__(self, input_embedding: Embedding, activation=None,
                 add_biases=False, projection_regularizer=None,
                 projection_dropout: float=0.0, **kwargs):
        self.embedding = input_embedding
        self.activation = activations.get(activation)
        self.add_biases = add_biases
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.projection_dropout = projection_dropout
        super().__init__(**kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.projection = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.embedding.output_dim),
            initializer='glorot_uniform',
            regularizer=self.projection_regularizer,
            trainable=True)
        if self.add_biases:
            self.biases = self.add_weight(
                name='biases',
                shape=(self.embedding.output_dim,),
                initializer='zeros',
                trainable=True)
            self.emb_biases = self.add_weight(
                name='emb_biases',
                shape=(self.embedding.input_dim,),
                initializer='zeros',
                trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape_tensor = K.shape(inputs)
        last_input_dim = K.int_shape(inputs)[-1]
        projected = K.dot(K.reshape(inputs, (-1, last_input_dim)),
                          self.projection)
        if 0 < self.projection_dropout < 1:
            projected = K.in_train_phase(
                lambda: K.dropout(projected, self.projection_dropout),
                projected,
                training=kwargs.get('training'))
        if self.add_biases:
            projected = K.bias_add(projected, self.biases,
                                   data_format='channels_last')
        attention = K.dot(projected, K.transpose(self.embedding.embeddings))
        if self.add_biases:
            attention = K.bias_add(attention, self.emb_biases,
                                   data_format='channels_last')
        result = K.reshape(
            self.activation(attention),
            (input_shape_tensor[0],
             input_shape_tensor[1],
             self.embedding.input_dim))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embedding.input_dim