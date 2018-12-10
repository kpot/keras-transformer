import argparse

import os
from itertools import islice
from typing import Iterable, List, Optional

from keras import optimizers, losses
from keras.models import load_model
# noinspection PyPep8Naming
from keras import backend as K
from keras import callbacks
import numpy as np

from . import wikitext
from .bpe import BPEEncoder, ID_FOR_PADDING
from .utils import (
    load_optimizer_weights, contain_tf_gpu_mem_usage, CosineLRSchedule)
from .models import (
    universal_transformer_gpt_model, vanilla_transformer_gpt_model)


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
    Reads WikiText dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = wikitext.read_wikitext_file(training_set_name)
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
    Reads WikiText dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = wikitext.read_wikitext_file(training_set_name)
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
        sample_sequence = [word_id for word_id, _ in chunk]
        useful_sequences.append(sample_sequence)

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result


def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


def main(model_save_path: str,
         model_name: str,
         tensorboard_log_path: Optional[str],
         num_epochs: int,
         learning_rate: float,
         batch_size: int,
         max_seq_length: int,
         word_embedding_size: int,
         load_weights_only: bool,
         show_model_summary: bool):
    contain_tf_gpu_mem_usage()
    encoder = wikitext.build_wikitext_bpe_encoder()

    def x_y_for_dataset(dataset_name):
        fat_sample = training_data_to_dense_samples(
            dataset_name, encoder, max_seq_length)
        _x = fat_sample[:, :max_seq_length]
        _y = np.expand_dims(fat_sample[:, 1:], axis=-1)
        return _x, _y

    x, y = x_y_for_dataset(wikitext.TRAINING_SET_NAME)

    def compile_new_model():
        if model_name == 'universal':
            optimizer = optimizers.Adam(
                lr=learning_rate, beta_1=0.6, beta_2=0.999)
            _model = universal_transformer_gpt_model(
                max_seq_length=max_seq_length,
                vocabulary_size=encoder.vocabulary_size(),
                word_embedding_size=word_embedding_size,
                transformer_depth=5,
                num_heads=8)
            _model.compile(
                optimizer,
                loss=losses.sparse_categorical_crossentropy,
                metrics=[perplexity])
        elif model_name == 'vanilla':
            optimizer = optimizers.Adam(
                lr=learning_rate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
            _model = vanilla_transformer_gpt_model(
                max_seq_length=max_seq_length,
                vocabulary_size=encoder.vocabulary_size(),
                word_embedding_size=word_embedding_size,
                transformer_depth=5,
                num_heads=8)
            _model.compile(
                optimizer,
                loss=losses.sparse_categorical_crossentropy,
                metrics=[perplexity])
        else:
            raise RuntimeError(f'Unknown model {model_name}')
        return _model

    if os.path.exists(model_save_path):
        if load_weights_only:
            print('Loading weights from', model_save_path)
            model = compile_new_model()
            model.load_weights(
                model_save_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_save_path)
        else:
            print('Loading the whole model from', model_save_path)
            model = load_model(
                model_save_path,
                custom_objects={
                    'perplexity': perplexity,
                })
    else:
        model = compile_new_model()

    if show_model_summary:
        model.summary(120)

    lr_scheduler = callbacks.LearningRateScheduler(
        CosineLRSchedule(lr_high=learning_rate,
                         lr_low=learning_rate / 32,
                         initial_period=num_epochs),
        verbose=1)
    model_callbacks = [
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss', save_best_only=True, verbose=True),
        lr_scheduler,
    ]
    if tensorboard_log_path:
        model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))
    model.fit(
        x, y,
        validation_data=x_y_for_dataset(wikitext.VALIDATION_SET_NAME),
        batch_size=batch_size, epochs=num_epochs,
        callbacks=model_callbacks)
    # Evaluation using test set
    print('-' * 80)
    test_x, test_y = x_y_for_dataset(wikitext.TEST_SET_NAME)
    test_metrics = model.evaluate(test_x, test_y, batch_size=batch_size)
    for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        print(f'Test {metric_name}:', metric_value)


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='A simple example of the Transformer model in work',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--save', type=str, required=True, metavar='PATH',
        help='A path where the best model should be saved / restored from')
    _argparser.add_argument(
        '--tensorboard-log', type=str, metavar='PATH', default=None,
        help='Path to a directory for Tensorboard logs')
    _argparser.add_argument(
        '--epochs', type=int, default=200, metavar='INTEGER',
        help='The number of epochs to train')
    _argparser.add_argument(
        '--lr', type=float, default=2e-4, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--batch-size', type=int, default=32, metavar='INTEGER',
        help='Training batch size')
    _argparser.add_argument(
        '--seq-len', type=int, default=256, metavar='INTEGER',
        help='Max sequence length')
    _argparser.add_argument(
        '--we-size', type=int, default=512, metavar='INTEGER',
        help='Word embedding size')
    _argparser.add_argument(
        '--model', type=str, default='universal', metavar='NAME',
        choices=['universal', 'vanilla'],
        help='The type of the model to train: "vanilla" or "universal"')
    _argparser.add_argument(
        '--load-weights-only', action='store_true',
        help='Use the save file only to initialize weights '
             '(do not load the whole model)')
    _argparser.add_argument(
        '--model-summary', action='store_true',
        help='Display the summary of the model before the training begins')
    _args = _argparser.parse_args()

    main(model_save_path=_args.save,
         model_name=_args.model,
         tensorboard_log_path=_args.tensorboard_log,
         num_epochs=_args.epochs,
         learning_rate=_args.lr,
         batch_size=_args.batch_size,
         max_seq_length=_args.seq_len,
         word_embedding_size=_args.we_size,
         load_weights_only=_args.load_weights_only,
         show_model_summary=_args.model_summary)
