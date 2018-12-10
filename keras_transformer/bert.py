"""
BERT stands for Bidirectional Encoder Representations from Transformers.

It's a way of pre-training Transformer to model a language, described in
paper [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/abs/1810.04805). A quote from it:

> BERT is designed to pre-train deep bidirectional representations
> by jointly conditioning on both left and right context in all layers.
> As a result, the pre-trained BERT representations can be fine-tuned
> with just one additional output layer to create state-of-the art
> models for a wide range of tasks, such as question answering
> and language inference, without substantial task-specific architecture
> modifications.
"""

import random
from itertools import islice, chain
from typing import List, Callable

import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects


class BatchGeneratorForBERT:
    """
    This class generates batches for a BERT-based language model
    in an abstract way, by using an external function sampling
    sequences of token IDs of a given length.
    """

    reserved_positions = 3

    def __init__(self, sampler: Callable[[int], List[int]],
                 dataset_size: int,
                 sep_token_id: int,
                 cls_token_id: int,
                 mask_token_id: int,
                 first_normal_token_id: int,
                 last_normal_token_id: int,
                 sequence_length: int,
                 batch_size: int,
                 sentence_min_span: float = 0.25):
        """
        :param sampler: A callable object responsible for uniformly sampling
            pieces of the dataset (already turned into token IDs).
            It should take one int argument - the sample length, and return
            a list of token IDs of the requested size.
        :param dataset_size: How big the whole dataset is, measured in number
            of token IDs.
        :param sep_token_id: ID of a token used as a separator between
            the sentences (called "[SEP]" in the paper).
        :param cls_token_id: ID of a token marking the node/position
            responsible for classification (always the first node).
            The token is called "[CLS]" in the original paper.
        :param mask_token_id: ID of a token masking the original words
            of the sentence, which the network should learn to "restore" using
            the context.
        :param first_normal_token_id: ID of the first token representing
            a normal word/token, not a specialized token, like "[SEP]".
        :param last_normal_token_id: ID of the last token representing
            a normal word, not a specialized token.
        :param sequence_length: a sequence length that can be accepted
            by the model being trained / validate.
        :param batch_size: how many samples each batch should include.
        :param sentence_min_span: A floating number ranging from 0 to 1,
            indicating the percentage of words (of the `sequence_length`)
            a shortest sentence should occupy. For example,
            if the value is 0.25, each sentence will vary in length from 25%
            to 75% of the whole `sequence_length` (minus 3 reserved positions
            for [CLS] and [SEP] tokens).
        """
        self.sampler = sampler
        self.steps_per_epoch = (
            # We sample the dataset randomly. So we can make only a crude
            # estimation of how many steps it should take to cover most of it.
            dataset_size // (sequence_length * batch_size))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.mask_token_id = mask_token_id
        self.first_token_id = first_normal_token_id
        self.last_token_id = last_normal_token_id
        assert 0.0 < sentence_min_span <= 1.0
        self.sentence_min_length = max(
            int(sentence_min_span *
                (self.sequence_length - self.reserved_positions)),
            1)
        self.sentence_max_length = (
            self.sequence_length
            - self.reserved_positions
            - self.sentence_min_length)

    def generate_batches(self):
        """
        Keras-compatible generator of batches for BERT (can be used with
        `keras.models.Model.fit_generator`).

        Generates tuples of (inputs, targets).
        `inputs` is a list of two values:
            1. masked_sequence: an integer tensor shaped as
               (batch_size, sequence_length), containing token ids of
               the input sequence, with some words masked by the [MASK] token.
            2. segment id: an integer tensor shaped as
               (batch_size, sequence_length),
               and containing 0 or 1 depending on which segment (A or B)
               each position is related to.

        `targets` is also a list of two values:
            1. combined_label: an integer tensor of a shape
               (batch_size, sequence_length, 2), containing both
               - the original token ids
               - and the mask (0s and 1s, indicating places where
                 a word has been replaced).
               both stacked along the last dimension.
               So combined_label[:, :, 0] would slice only the token ids,
               and combined_label[:, :, 1] would slice only the mask.
            2. has_next: a float32 tensor (batch_size, 1) containing
               1s for all samples where "sentence B" is directly following
               the "sentence A", and 0s otherwise.
        """
        samples = self.generate_samples()
        while True:
            next_bunch_of_samples = islice(samples, self.batch_size)
            has_next, mask, sequence, segment, masked_sequence = zip(
                *list(next_bunch_of_samples))
            combined_label = np.stack([sequence, mask], axis=-1)
            yield (
                [np.array(masked_sequence), np.array(segment)],
                [combined_label,
                 np.expand_dims(np.array(has_next, dtype=np.float32), axis=-1)]
            )

    def generate_samples(self):
        """
        Generates samples, one by one, for later concatenation into batches
        by `generate_batches()`.
        """
        while True:
            # Sentence A has length between 25% and 75% of the whole sequence
            a_length = random.randint(
                self.sentence_min_length,
                self.sentence_max_length)
            b_length = (
               self.sequence_length - self.reserved_positions - a_length)

            # Sampling sentences A and B,
            # making sure they follow each other 50% of the time
            has_next = random.random() < 0.5
            if has_next:
                # sentence B is a continuation of A
                full_sample = self.sampler(a_length + b_length)
                sentence_a = full_sample[:a_length]
                sentence_b = full_sample[a_length:]
            else:
                # sentence B is not a continuation of A
                # note that in theory the same or overlapping sentence
                # can be selected as B, but it's highly improbable
                # and shouldn't affect the performance
                sentence_a = self.sampler(a_length)
                sentence_b = self.sampler(b_length)
            assert len(sentence_a) == a_length
            assert len(sentence_b) == b_length
            sequence = (
                [self.cls_token_id] +
                sentence_a + [self.sep_token_id] +
                sentence_b + [self.sep_token_id])
            masked_sequence = sequence.copy()
            output_mask = np.zeros((len(sequence),), dtype=int)
            segment_id = np.full((len(sequence),), 1, dtype=int)
            segment_id[:a_length + 2] = 0
            for word_pos in chain(
                    range(1, a_length + 1),
                    range(a_length + 2, a_length + 2 + b_length)):
                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_sequence[word_pos] = self.mask_token_id
                    elif dice < 0.9:
                        masked_sequence[word_pos] = random.randint(
                            self.first_token_id, self.last_token_id)
                    # else: 10% of the time we just leave the word as is
                    output_mask[word_pos] = 1
            yield (int(has_next), output_mask, sequence,
                   segment_id, masked_sequence)


def masked_perplexity(y_true, y_pred):
    """
    Masked version of popular metric for evaluating performance of
    language modelling architectures. It assumes that y_pred has shape
    (batch_size, sequence_length, 2), containing both
      - the original token ids
      - and the mask (0s and 1s, indicating places where
        a word has been replaced).
    both stacked along the last dimension.
    Masked perplexity ignores all but masked words.

    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    y_true_value = y_true[:, :, 0]
    mask = y_true[:, :, 1]
    cross_entropy = K.sparse_categorical_crossentropy(y_true_value, y_pred)
    batch_perplexities = K.exp(
        K.sum(mask * cross_entropy, axis=-1) / (K.sum(mask, axis=-1) + 1e-6))
    return K.mean(batch_perplexities)


class MaskedPenalizedSparseCategoricalCrossentropy:
    """
    Masked cross-entropy (see `masked_perplexity` for more details)
    loss function with penalized confidence.
    Combines two loss functions: cross-entropy and negative entropy
    (weighted by `penalty_weight` parameter), following paper
    "Regularizing Neural Networks by Penalizing Confident Output Distributions"
    (https://arxiv.org/abs/1701.06548)

    how to use:
    >>> model.compile(
    >>>     optimizer,
    >>>     loss=MaskedPenalizedSparseCategoricalCrossentropy(0.1))
    """
    def __init__(self, penalty_weight: float):
        self.penalty_weight = penalty_weight

    def __call__(self, y_true, y_pred):
        y_true_val = y_true[:, :, 0]
        mask = y_true[:, :, 1]

        # masked per-sample means of each loss
        num_items_masked = K.sum(mask, axis=-1) + 1e-6
        masked_cross_entropy = (
            K.sum(mask * K.sparse_categorical_crossentropy(y_true_val, y_pred),
                  axis=-1)
            / num_items_masked)
        masked_entropy = (
            K.sum(mask * -K.sum(y_pred * K.log(y_pred), axis=-1), axis=-1)
            / num_items_masked)
        return masked_cross_entropy - self.penalty_weight * masked_entropy

    def get_config(self):
        return {
            'penalty_weight': self.penalty_weight
        }


get_custom_objects().update({
    'MaskedPenalizedSparseCategoricalCrossentropy':
        MaskedPenalizedSparseCategoricalCrossentropy,
    'masked_perplexity': masked_perplexity,
})
