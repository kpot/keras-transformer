import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects


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
        num_dims = len(input_shape) - 2
        channels = input_shape[-1]
        num_timescales = channels // (num_dims * 2)
        log_timescale_increment = K.constant(
            np.log(float(self.max_timescale) / float(self.min_timescale)) /
            (num_timescales - 1),
            dtype=K.floatx())
        inv_timescales = self.min_timescale * K.exp(
            K.arange(num_timescales, dtype=K.floatx()) * -log_timescale_increment)
        self.signals = []
        for dim in range(num_dims):
            length = input_shape[dim + 1]
            position = K.arange(length, dtype=K.floatx())
            scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
            signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            padded = [signal]
            if prepad:
                padded.insert(0, K.zeros((length, prepad)))
            if postpad:
                padded.append(K.zeros((length, postpad)))
            signal = K.concatenate(padded, 1)
            for _ in range(1 + dim):
                signal = K.expand_dims(signal, 0)
            for _ in range(num_dims - 1 - dim):
                signal = K.expand_dims(signal, -2)
            self.signals.append(signal)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        for signal in self.signals:
            inputs = inputs + signal
        return inputs


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
