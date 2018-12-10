import argparse
from typing import Optional
import random
import os

from keras.models import load_model
# noinspection PyPep8Naming
from keras import optimizers
from keras import callbacks
from keras import losses

from keras_transformer.bert import (
    BatchGeneratorForBERT, masked_perplexity,
    MaskedPenalizedSparseCategoricalCrossentropy)

from . import wikitext
from .bpe import BPEEncoder
from .utils import (
    load_optimizer_weights, contain_tf_gpu_mem_usage, CosineLRSchedule)
from .models import transformer_bert_model

BERT_SPECIAL_TOKENS = ['[SEP]', '[CLS]', '[MASK]']

# Penalty for confidence of the output distribution, as described in
# "Regularizing Neural Networks by Penalizing Confident Output Distributions"
# (https://arxiv.org/abs/1701.06548)
CONFIDENCE_PENALTY = 0.1


def stream_bpe_token_ids(text: str, encoder: BPEEncoder):
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        # the encoder is supposed to add <SEQ> and </SEQ>
        for token_id, token in encoder(clean_line):
            yield token_id


def wikitext_bert_generator(
        dataset_name: str, encoder: BPEEncoder,
        batch_size: int, sequence_length: int) -> BatchGeneratorForBERT:
    text = wikitext.read_wikitext_file(dataset_name)
    token_ids = list(stream_bpe_token_ids(text, encoder))

    def sampler(size):
        start = random.randint(0, len(token_ids) - size - 1)
        return token_ids[start: start + size]

    sep_token_id, cls_token_id, mask_token_id = [
        encoder.vocabulary.token_to_id[token]
        for token in BERT_SPECIAL_TOKENS]
    generator = BatchGeneratorForBERT(
        sampler=sampler,
        dataset_size=len(token_ids),
        sep_token_id=sep_token_id,
        cls_token_id=cls_token_id,
        mask_token_id=mask_token_id,
        first_normal_token_id=encoder.vocabulary.first_normal_token_id,
        last_normal_token_id=encoder.vocabulary.last_normal_token_id,
        sequence_length=sequence_length,
        batch_size=batch_size)
    return generator


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
    encoder = wikitext.build_wikitext_bpe_encoder(
        special_tokens=BERT_SPECIAL_TOKENS)

    def compile_new_model():
        optimizer = optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999)
        _model = transformer_bert_model(
            use_universal_transformer=(model_name == 'universal'),
            max_seq_length=max_seq_length,
            vocabulary_size=encoder.vocabulary_size(),
            word_embedding_size=word_embedding_size,
            transformer_depth=5,
            num_heads=8)
        _model.compile(
            optimizer,
            loss=[
                MaskedPenalizedSparseCategoricalCrossentropy(
                    CONFIDENCE_PENALTY),
                losses.binary_crossentropy],
            metrics={'word_predictions': masked_perplexity})
        return _model

    if os.path.exists(model_save_path):
        if load_weights_only:
            print('Loading weights from', model_save_path)
            model = compile_new_model()
            model.load_weights(model_save_path,
                               skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_save_path)
        else:
            print('Loading the whole model from', model_save_path)
            model = load_model(
                model_save_path,
                custom_objects={
                    'masked_perplexity': masked_perplexity,
                })
    else:
        model = compile_new_model()

    if show_model_summary:
        model.summary(120)

    lr_scheduler = callbacks.LearningRateScheduler(
        CosineLRSchedule(lr_high=learning_rate, lr_low=1e-8,
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

    training_batches = wikitext_bert_generator(
        wikitext.TRAINING_SET_NAME, encoder, batch_size, max_seq_length)
    validation_batches = wikitext_bert_generator(
        wikitext.VALIDATION_SET_NAME, encoder, batch_size, max_seq_length)
    model.fit_generator(
        generator=training_batches.generate_batches(),
        steps_per_epoch=training_batches.steps_per_epoch,
        epochs=num_epochs,
        callbacks=model_callbacks,
        validation_data=validation_batches.generate_batches(),
        validation_steps=validation_batches.steps_per_epoch,
    )
    # Evaluation using test set
    print('-' * 80)
    test_batches = wikitext_bert_generator(
        wikitext.TEST_SET_NAME, encoder, batch_size, max_seq_length)
    test_metrics = model.evaluate_generator(
        test_batches.generate_batches(),
        test_batches.steps_per_epoch)
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
        '--epochs', type=int, default=1000, metavar='INTEGER',
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
