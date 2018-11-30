import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects


def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


class AddCoordinateEncoding(AddPositionalEncoding):
    """
    Implements coordinate encoding described in section 2.1
    of "Universal Transformers" (https://arxiv.org/abs/1807.03819).
    In other words, injects two signals at once: current position in
    the sequence, and current step (vertically) in the transformer model.
    """

    def build(self, input_shape):
        super().build(input_shape)
        _, length, hidden_size = input_shape

    def call(self, inputs, step=None, **kwargs):
        if step is None:
            raise ValueError("Please, provide current Transformer's step"
                             "using 'step' keyword argument.")
        pos_encoded_added = super().call(inputs, **kwargs)
        step_signal = K.expand_dims(self.signal[:, step, :], axis=1)
        return pos_encoded_added + step_signal


class TransformerCoordinateEmbedding(Layer):
    """
    Represents trainable positional embeddings for the Transformer model:

    1. word position embeddings - one for each position in the sequence.
    2. depth embeddings - one for each block of the model

    Calling the layer with the Transformer's input will return a new input
    with those embeddings added.
    """

    def __init__(self, max_transformer_depth: int, **kwargs):
        self.max_depth = max_transformer_depth
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['max_transformer_depth'] = self.max_depth
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        sequence_length, d_model = input_shape[-2:]
        self.word_position_embeddings = self.add_weight(
            shape=(sequence_length, d_model),
            initializer='uniform',
            name='word_position_embeddings',
            trainable=True)
        self.depth_embeddings = self.add_weight(
            shape=(self.max_depth, d_model),
            initializer='uniform',
            name='depth_position_embeddings',
            trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        depth = kwargs.get('step')
        if depth is None:
            raise ValueError("Please, provide current Transformer's step"
                             "using 'step' keyword argument.")
        result = inputs + self.word_position_embeddings
        if depth is not None:
            result = result + self.depth_embeddings[depth]
        return result


get_custom_objects().update({
    'TransformerCoordinateEmbedding': TransformerCoordinateEmbedding,
    'AddCoordinateEncoding': AddCoordinateEncoding,
    'AddPositionalEncoding': AddCoordinateEncoding,
})
