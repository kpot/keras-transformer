import collections
from typing import (
    Iterable, Tuple, List, NamedTuple, Dict, Sequence, FrozenSet, TextIO,
    Callable, Optional)


BPE_WORD_TAIL = '</w>'

MAX_MERGE_RANK = 0xffffffff
TOKEN_FOR_NUMBERS = '<NUM>'
TOKEN_FOR_UNKNOWN = '<UNK>'
TOKEN_FOR_BEGINNING_OF_SEQUENCE = '<SEQ>'
TOKEN_FOR_END_OF_SEQUENCE = '</SEQ>'
TOKEN_FOR_PADDING = '<PAD>'
ID_FOR_UNKNOWN_TOKEN = 0
ID_FOR_BEGINNING_OF_SEQUENCE = 1
ID_FOR_END_OF_SEQUENCE = 2
ID_FOR_PADDING = 3


class BPEMerges(NamedTuple):
    # the full list of merges in the order of their appearance in the file
    merges: List[Tuple[str, str]]
    # merges mapped to their positions in the file
    ranks: Dict[Tuple[str, str], int]

    @staticmethod
    def pack(merges: List[Tuple[str, str]]) -> 'BPEMerges':
        ranks = {m: i for i, m in enumerate(merges)}
        return BPEMerges(merges, ranks)

    @staticmethod
    def load_from_file(file: TextIO) -> 'BPEMerges':
        merges = []
        for line in file:
            if line.startswith('#'):
                continue
            first, second = line.strip().split()
            merges.append((first, second))
        if len(merges) >= MAX_MERGE_RANK:
            raise ValueError('Too many merges')
        return BPEMerges.pack(merges)

    @staticmethod
    def load(file_path: str) -> 'BPEMerges':
        with open(file_path, 'rt', encoding='utf-8') as f:
            return BPEMerges.load_from_file(f)


def pairs_of_symbols(word: Sequence[str]) -> FrozenSet[Tuple[str, str]]:
    return frozenset(zip(word, word[1:]))


def apply_bpe(token: str, merges: BPEMerges) -> Tuple[str, ...]:
    """
    Takes a string and converts it into a tuple of strings each of which
    representing a single BPE token (after all transformations applied).
    :param token: any string
    :param merges: list of merging operations (see docs for more info on that)
    :return: tuple of strings
    """
    word = tuple(token[:-1]) + (token[-1] + BPE_WORD_TAIL,)  # type: tuple
    if len(word) == 1:
        # a single-character text
        return (token + BPE_WORD_TAIL,)
    pairs = pairs_of_symbols(word)
    while True:
        bigram = min(pairs,
                     key=lambda pair: merges.ranks.get(pair, MAX_MERGE_RANK))
        if bigram not in merges.ranks:
            # we have no merges left to follow
            break
        # We have a bigram that needs to be merged. Now we need to identify
        # all occurrences of this bigram in the token and merge them
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                # looking for the first symbol of the bigram
                j = word.index(first, i)
            except ValueError:
                # No more occurrences. Just copy the rest of the token and stop
                new_word.extend(word[i:])
                break
            else:
                # We have a match, which doesn't mean it's a full bigram yet
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    # it's a full bigram, performing the merge
                    new_word.append(first + second)
                    i += 2
                else:
                    # it's just the first symbol, simply copy it and continue
                    new_word.append(word[i])
                    i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = pairs_of_symbols(word)
    return word


class BPETokenizer:
    """
    Converts given text like "Take 2 spoons" into a stream of BPE-processed
    tokens looking like ("tak", "e</w>", NumToken(...), "spoon", "s</w>").
    """

    def __init__(self, merges: BPEMerges,
                 word_tokenizer: Callable[[str], Iterable[str]],
                 mark_sequence_edges: bool = True):
        self.word_tokenizer = word_tokenizer
        self.merges = merges
        self.bpe_cache = {}
        self.mark_sequence_edges = mark_sequence_edges

    def apply(self, text: str,
              low_case: bool = True) -> Iterable[str]:
        if self.mark_sequence_edges:
            yield TOKEN_FOR_BEGINNING_OF_SEQUENCE
        for token in self.word_tokenizer(text):
            to_parse = token.lower() if low_case else token
            try:
                encoded_tokens = self.bpe_cache[to_parse]
            except KeyError:
                encoded_tokens = apply_bpe(to_parse, self.merges)
                self.bpe_cache[to_parse] = encoded_tokens
            yield from encoded_tokens
        if self.mark_sequence_edges:
            yield TOKEN_FOR_END_OF_SEQUENCE


class BPEVocabulary:
    def __init__(self, bpe_vocabulary_file: TextIO,
                 special_tokens: Optional[Sequence[str]] = None):
        vocabulary = {
            TOKEN_FOR_UNKNOWN: ID_FOR_UNKNOWN_TOKEN,
            TOKEN_FOR_BEGINNING_OF_SEQUENCE: ID_FOR_BEGINNING_OF_SEQUENCE,
            TOKEN_FOR_END_OF_SEQUENCE: ID_FOR_END_OF_SEQUENCE,
            TOKEN_FOR_PADDING: ID_FOR_PADDING}
        i = max(vocabulary.values()) + 1
        if special_tokens is not None:
            for extra_token in special_tokens:
                if extra_token not in vocabulary:
                    vocabulary[extra_token] = i
                    i += 1
        self.first_normal_token_id = i
        assert i == len(vocabulary)
        for line in bpe_vocabulary_file:
            if line:
                word, frequency = line.strip().rsplit(' ')
                if word not in vocabulary:
                    vocabulary[word] = i
                    i += 1
        self.token_to_id = vocabulary
        self.id_to_token = {v: k for k, v in vocabulary.items()}
        self.last_normal_token_id = i - 1


class BPEEncoder:
    """
    Converts a text into a stream of WordIDs and BPE tokens
    """
    def __init__(self, bpe_tokenizer: BPETokenizer, bpe_vocabulary: TextIO,
                 special_tokens: Optional[Sequence[str]] = None):
        self.bpe_tokenizer = bpe_tokenizer
        self.vocabulary = BPEVocabulary(
            bpe_vocabulary, special_tokens=special_tokens)

    def __call__(self, text: str) -> Iterable[Tuple[int, str]]:
        token_to_id = self.vocabulary.token_to_id
        for token in self.bpe_tokenizer.apply(text):
            token_str = str(token)
            yield (token_to_id.get(token_str, ID_FOR_UNKNOWN_TOKEN), token)

    def vocabulary_size(self):
        return len(self.vocabulary.token_to_id)


def build_vocabulary(tokens: Iterable[str]) -> List[Tuple[str, int]]:
    vocabulary = collections.defaultdict(int)
    for token in tokens:
        vocabulary[token.lower()] += 1
    return sorted(vocabulary.items(), key=lambda i: (i[1], i[0]), reverse=True)
