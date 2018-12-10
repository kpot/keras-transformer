import zipfile
from typing import Callable, Optional, Sequence, Iterable
import os
import io
import re

from subword_nmt.learn_bpe import learn_bpe
import tqdm

from .bpe import (
    BPEEncoder, TOKEN_FOR_NUMBERS, build_vocabulary, BPETokenizer, BPEMerges)
from .tokenizer import RegexTokenizer


NUM_BPE_MERGES = 10000

WIKITEXT_WORD_LEVEL = True

WIKITEXT_ZIP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'wikitext-2-v1.zip' if WIKITEXT_WORD_LEVEL else 'wikitext-2-raw-v1.zip')

TRAINING_SET_NAME = (
    'wikitext-2/wiki.train.tokens' if WIKITEXT_WORD_LEVEL
    else 'wikitext-2-raw/wiki.train.raw')
VALIDATION_SET_NAME = (
    'wikitext-2/wiki.valid.tokens' if WIKITEXT_WORD_LEVEL
    else 'wikitext-2-raw/wiki.valid.raw')
TEST_SET_NAME = (
    'wikitext-2/wiki.test.tokens' if WIKITEXT_WORD_LEVEL
    else 'wikitext-2-raw/wiki.test.raw')


def read_wikitext_file(file_name):
    z = zipfile.ZipFile(WIKITEXT_ZIP)
    text = z.read(file_name).decode('utf-8')
    text = re.sub(r'\s@(.)@\s', r'\1', text)
    return text


def build_wikitext_bpe_encoder(
        special_tokens: Optional[Sequence[str]] = None) -> BPEEncoder:

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
        train_tokens = read_wikitext_file(TRAINING_SET_NAME)
        all_lines = train_tokens.splitlines()
        for line in tqdm.tqdm(all_lines, desc=description):
            yield from tokenizer(line)

    vocabulary_file = io.StringIO(
        '\n'.join(
            '{} {}'.format(word, counter)
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
    bpe_vocabulary = build_vocabulary(
        wikitext_tokens(bpe_tokenizer.apply, 'Building BPE vocabulary'))
    print('BPE Vocabulary size:', len(bpe_vocabulary))
    bpe_vocabulary_file = io.StringIO(
        '\n'.join('{} {}'.format(word, counter)
                  for word, counter in bpe_vocabulary))
    bpe_encoder = BPEEncoder(bpe_tokenizer, bpe_vocabulary_file,
                             special_tokens=special_tokens)

    return bpe_encoder
