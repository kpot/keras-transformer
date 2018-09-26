import zipfile
import os
import io
from itertools import islice
from typing import Iterable, Callable, List, Optional

import tqdm
from keras import Input, optimizers, losses, metrics
from keras.models import Model
from keras.layers import Embedding, Layer, Softmax, Dropout, regularizers
from keras import activations
from keras import backend as K
from keras import callbacks
from subword_nmt.learn_bpe import learn_bpe
import numpy as np

from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerBlock, TransformerACT
from .bpe import (
    build_vocabulary, TOKEN_FOR_NUMBERS, BPEEncoder, BPETokenizer, BPEMerges,
    ID_FOR_PADDING, ID_FOR_BEGINNING_OF_SEQUENCE, ID_FOR_END_OF_SEQUENCE)
from .tokenizer import RegexTokenizer

NUM_BPE_MERGES = 10000
MAX_SEQ_LENGTH = 256
WORD_EMBEDDING_SIZE = 512

WIKITEXT_ZIP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'wikitext-2-v1.zip')

WIKITEXT_TRAINING_SET_NAME = 'wikitext-2/wiki.train.tokens'
WIKITEXT_VALIDATION_SET_NAME = 'wikitext-2/wiki.valid.tokens'
WIKITEXT_TEST_SET_NAME = 'wikitext-2/wiki.test.tokens'


def read_wikitext_file(file_name):
    z = zipfile.ZipFile(WIKITEXT_ZIP)
    train_tokens = z.read(file_name).decode('utf-8')
    return train_tokens


def build_wikitext_bpe_encoder() -> BPEEncoder:

    regex_tokenizer = RegexTokenizer()

    def tokenize_with_regex(text: str) -> Iterable[str]:
        document = regex_tokenizer.apply(text)
        for token in document:
            if token.number:
                yield TOKEN_FOR_NUMBERS
            else:
                yield str(token)

    def wikitext_tokens(tokenizer: Callable[[str], Iterable[str]],
                        description: str):
        train_tokens = read_wikitext_file(WIKITEXT_TRAINING_SET_NAME)
        all_lines = train_tokens.splitlines()
        for line in tqdm.tqdm(all_lines, desc=description):
            yield from tokenizer(line)

    vocabulary_file = io.StringIO(
        '\n'.join(
            f'{word} {counter}'
            for word, counter in build_vocabulary(
                wikitext_tokens(tokenize_with_regex,
                                'Building vocabulary'))))

    with io.StringIO() as file_with_merges:
        print('Learning BPE...', flush=True, end='')
        learn_bpe(vocabulary_file, file_with_merges, NUM_BPE_MERGES,
                  min_frequency=3, verbose=False, is_dict=True,
                  total_symbols=False)
        file_with_merges.seek(0)
        print('Done', flush=True)
        merges = BPEMerges.load_from_file(file_with_merges)

    bpe_tokenizer = BPETokenizer(
        merges, tokenize_with_regex, mark_sequence_edges=True)
    bpe_vocabulary_file = io.StringIO(
        '\n'.join(
            f'{word} {counter}'
            for word, counter in build_vocabulary(
                wikitext_tokens(bpe_tokenizer.apply,
                                'Building BPE vocabulary'))))
    bpe_encoder = BPEEncoder(bpe_tokenizer, bpe_vocabulary_file)

    return bpe_encoder


def pad_lm_samples(samples: Iterable[List[int]],
                   required_sequence_length: int):
    tail_padding = [ID_FOR_PADDING]
    for sample in samples:
        assert len(sample) > 0
        sample.extend(tail_padding * (required_sequence_length - len(sample)))


def training_data_to_samples(training_set_name: str,
                             encoder: BPEEncoder,
                             max_sequence_length: int) -> np.ndarray:
    """
    Reads wikitext dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = read_wikitext_file(training_set_name)
    useful_sequences = []
    for line in training_set.splitlines():
        clean_line = line.strip()
        is_header = clean_line.startswith('=') and clean_line.endswith('=')
        if is_header or not clean_line:
            continue
        # the encoder is supposed to add <SEQ> and </SEQ>
        id_word_pairs = list(encoder(clean_line))
        useful_sequences.append(
            [word_id for word_id, _ in id_word_pairs[:max_sequence_length]])

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result


def training_data_to_dense_samples(training_set_name: str,
                                   encoder: BPEEncoder,
                                   max_sequence_length: int) -> np.ndarray:
    """
    Reads wikitext dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = read_wikitext_file(training_set_name)
    useful_sequences = []

    def stream_bpe_tokens():
        for line in training_set.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            # the encoder is supposed to add <SEQ> and </SEQ>
            id_word_pairs = encoder(clean_line)
            yield from id_word_pairs

    id_word_stream = stream_bpe_tokens()
    while True:
        chunk = list(islice(id_word_stream, max_sequence_length))
        if len(chunk) == 0:
            break
        # print(' '.join(token for _, token in chunk))
        sample_sequence = [word_id for word_id, _ in chunk]
        useful_sequences.append(sample_sequence)

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result



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
                 add_biases=False, **kwargs):
        self.embedding = input_embedding
        self.activation = activations.get(activation)
        self.add_biases = add_biases
        super().__init__(**kwargs)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.projection = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.embedding.output_dim),
            initializer='glorot_uniform',
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


def make_transformer_model(max_seq_length: int, vocabulary_size: int,
                           word_embedding_size: int, transformer_depth: int,
                           num_heads: int, dropout_rate: float=0.1):
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    embedding_layer = Embedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=regularizers.l2(1e-4))
    output_layer = TiedOutputEmbedding(embedding_layer, name='word_prediction')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')
    transformer_act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = TransformerBlock(
        name='transformer', num_heads=num_heads, residual_dropout=dropout_rate,
        attention_dropout=dropout_rate, use_masking=True)

    next_step_input = act_output = embedding_layer(word_ids)
    dropout_layer = Dropout(dropout_rate, name='input_dropout')

    for i in range(transformer_depth):
        next_step_input = coordinate_embedding_layer(next_step_input, step=i)
        next_step_input = dropout_layer(next_step_input)
        next_step_input = transformer_block(next_step_input)
        next_step_input, act_output = transformer_act_layer(next_step_input)

    transformer_act_layer.finalize()
    next_step_input = act_output
    word_predictions = output_layer(next_step_input)
    return Model(inputs=[word_ids], outputs=[word_predictions])


def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


def sparse_categorical_crossentropy_with_logits(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def main(model_save_path: str,
         tensorboard_log_path: Optional[str],
         num_epochs: int,
         learning_rate: float,
         batch_size: int):
    encoder = build_wikitext_bpe_encoder()

    def x_y_for_dataset(dataset_name):
        fat_sample = training_data_to_dense_samples(
            dataset_name, encoder, MAX_SEQ_LENGTH)
        _x = fat_sample[:, :MAX_SEQ_LENGTH]
        _y = np.expand_dims(fat_sample[:, 1:], axis=-1)
        return _x, _y

    x, y = x_y_for_dataset(WIKITEXT_TRAINING_SET_NAME)
    model = make_transformer_model(
        max_seq_length=MAX_SEQ_LENGTH,
        vocabulary_size=encoder.vocabulary_size(),
        word_embedding_size=WORD_EMBEDDING_SIZE,
        transformer_depth=5,
        num_heads=8)
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer,
        loss=sparse_categorical_crossentropy_with_logits,
        metrics=[perplexity])
    if os.path.exists(model_save_path):
        print('Loading weights from', model_save_path)
        model.load_weights(model_save_path)
    model_callbacks = [
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_perplexity', save_best_only=True, verbose=True)
    ]
    if tensorboard_log_path:
        model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))
    model.fit(
        x, y,
        validation_data=x_y_for_dataset(WIKITEXT_VALIDATION_SET_NAME),
        batch_size=batch_size, epochs=num_epochs,
        callbacks=model_callbacks)
    test_x, test_y = x_y_for_dataset(WIKITEXT_TEST_SET_NAME)
    test_loss, test_perplexity = model.evaluate(
        test_x, test_y, batch_size=batch_size)
    print('Test loss:', test_loss)
    print('Test perplexity:', test_perplexity)


if __name__ == '__main__':
    main(model_save_path='lm_model.h5',
         tensorboard_log_path='tensorboard_log',
         num_epochs=50,
         learning_rate=2e-4,
         batch_size=64)
