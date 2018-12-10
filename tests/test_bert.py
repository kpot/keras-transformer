import random
from itertools import islice

import numpy as np
from keras_transformer.bert import BatchGeneratorForBERT


def test_bert_sample_generator():
    token_ids = list(range(3, 1000))

    def sampler(size):
        start = random.randint(0, len(token_ids) - size - 1)
        return token_ids[start: start + size]

    gen = BatchGeneratorForBERT(
        sampler, len(token_ids), sep_token_id=0, cls_token_id=1,
        mask_token_id=2, first_normal_token_id=3,
        last_normal_token_id=token_ids[-1],
        sequence_length=128, batch_size=16)
    for has_next, output_mask, sequence, section_id, masked_sequence in islice(
            gen.generate_samples(), 10):
        assert sequence[0] == gen.cls_token_id
        assert sequence[-1] == gen.sep_token_id
        assert len(sequence) == gen.sequence_length
        assert masked_sequence != sequence
        assert len(section_id) == gen.sequence_length

        assert np.sum(section_id == 0) > 1
        assert np.sum(section_id == 1) > 1
        assert (np.sum(section_id == 1) + np.sum(section_id == 0)
                == gen.sequence_length)

        first_sep = sequence.index(gen.sep_token_id)
        if has_next:
            # checking that the second sentence is truly a continuation
            assert sequence[first_sep - 1] == sequence[first_sep + 1] - 1
        else:
            assert sequence[first_sep - 1] != sequence[first_sep + 1] - 1
        # Checking that output_mask correctly marks the changes
        for i, (s, ms) in enumerate(zip(sequence, masked_sequence)):
            if s != ms:
                assert output_mask[i] == 1
            if output_mask[i] == 0:
                assert s == ms
            else:
                assert ms not in (gen.cls_token_id, gen.sep_token_id)

    # Checking batch generator
    batches = gen.generate_batches()
    batch = next(batches)
    x, y = batch
    assert isinstance(x[0], np.ndarray)
    assert isinstance(x[1], np.ndarray)
    assert x[0].shape == (gen.batch_size, gen.sequence_length)
    assert x[1].shape == (gen.batch_size, gen.sequence_length)
    assert isinstance(y, list)
    assert isinstance(y[0], np.ndarray)
    assert y[0].shape == (gen.batch_size, gen.sequence_length, 2)
    assert len(y[1]) == gen.batch_size
