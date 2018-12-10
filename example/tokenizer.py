"""
A simple tokenizer capable of extracting not only the words, but also both
numbers and ranges of them, automatically converting the last ones into
a universal format.

The main arguable disadvantage of this tokenizer is that it's unable to see
any tokens that don't match the regular expressions describing of what
a token should look like.
"""
from typing import NamedTuple, Optional, Tuple, Iterable
from fractions import Fraction
import re

import numpy as np

# A mapping between UNICODE fractions and their actual values
FRACTIONS_MAP = {
    '\u00bc': Fraction(1, 4),
    '\u00bd': Fraction(1, 2),
    '\u00be': Fraction(3, 4),
    '\u2150': Fraction(1, 7),
    '\u2151': Fraction(1, 9),
    '\u2152': Fraction(1, 10),
    '\u2153': Fraction(1, 3),
    '\u2154': Fraction(2, 3),
    '\u2155': Fraction(1, 5),
    '\u2156': Fraction(2, 5),
    '\u2157': Fraction(3, 5),
    '\u2158': Fraction(4, 5),
    '\u2159': Fraction(1, 6),
    '\u215a': Fraction(5, 6),
    '\u215b': Fraction(1, 8),
    '\u215c': Fraction(3, 8),
    '\u215d': Fraction(5, 8),
    '\u215e': Fraction(7, 8),
}

UNICODE_SUPERSCRIPT = ''.join((
    '\u2070',  # 0
    '\u00b9',  # 1
    '\u00b2',  # 2
    '\u00b3',  # 3
    '\u2074',  # 4
    '\u2075',  # 5
    '\u2076',  # 6
    '\u2077',  # 7
    '\u2078',  # 8
    '\u2079',  # 9
))

UNICODE_SUBSCRIPT = ''.join((
    '\u2080',  # 0
    '\u2081',  # 1
    '\u2082',  # 2
    '\u2083',  # 3
    '\u2084',  # 4
    '\u2085',  # 5
    '\u2086',  # 6
    '\u2087',  # 7
    '\u2088',  # 8
    '\u2089',  # 9
))

FRACTION_NUMERATOR_ONE = '\u215f'  # symbol "1/" (⅟)
SLASHES = re.escape('/\u2044')

UNICODE_FRACTIONS = re.escape(''.join(FRACTIONS_MAP.keys()))

# These regular expressions support only positive
NUMBER_RE_PIECES = [
    ('integer_per_cent', r'\d+%'),  # 15%
    ('floating_per_cent1', r'\d+\.\d+%'),  # 15.2%
    ('floating_per_cent2', r'\d+\,\d+%'),  # 15,2%
    ('floating_point1', r'\d+\.\d+'),  # 1.2 or 2,5
    ('floating_point2', r'\d+,\d+'),  # 1.2 or 2,5
    ('mixed_fraction1', rf'\d+\s+\d+\s*[{SLASHES}]\s*\d+'),  # 12 1/2
    ('mixed_fraction2', rf'\d+\s*[{UNICODE_FRACTIONS}]'),  # 1 ½
    ('mixed_fraction3', rf'\d+\s*[{UNICODE_SUPERSCRIPT}]+\s*'
                        rf'[{SLASHES}]\s*[{UNICODE_SUBSCRIPT}]+'),  # 1 ¹¹⁄₂₀
    ('common_fraction1', rf'\d+\s*[{SLASHES}]\s*\d+'),  # 3 / 4
    ('common_fraction2', rf'[{UNICODE_FRACTIONS}]'),  # ½
    ('common_fraction3', rf'{FRACTION_NUMERATOR_ONE}\s*\d+'),  # 1/...
    ('common_fraction4',  # ¹¹⁄₂₀
     rf'[{UNICODE_SUPERSCRIPT}]+\s*[{SLASHES}]\s*[{UNICODE_SUBSCRIPT}]+'),
    ('integer', r'\d+'),  # 1234
]

FRACTION_SEPARATOR = re.compile(rf'[\s{SLASHES}]+')
NUMBER_RE = r'(?:' + '|'.join(v for k, v in NUMBER_RE_PIECES) + ')'
NUMBER_RE_GROUPED = re.compile(
    '(?:' + '|'.join(rf'(?P<{k}>{v})' for k, v in NUMBER_RE_PIECES) + ')')
PUNCTUATION_RE = re.escape('\\.,:;{}()[]\\?!\'"\n\t/#*|↑~+')
DELIMITERS_RE = rf"(?:\.\.\.|[{PUNCTUATION_RE}])"


MAX_NUMBER_SCALE = 20
NUM_VECTORS_TEMPLATE = np.concatenate(
    [np.zeros((MAX_NUMBER_SCALE, 2),  # 2 for NumToken.{mean,span} fields
              dtype=np.float32),
     np.eye(MAX_NUMBER_SCALE, dtype=np.float32)],
    axis=1)

# A constant vector representing "Not a number" state
NAN_VECTOR = NUM_VECTORS_TEMPLATE[-1].copy()
NAN_VECTOR.flags.writeable = False


class Token(NamedTuple):
    # The area of the original text occupied by token (start and end positions)
    span: Tuple[int, int]
    # Tokens-delimiters have this field filled
    delimiter: Optional[str] = None
    # Tokens-words have this field filled
    word: Optional[str] = None
    # Tokens-numbers have this field filled
    number: Optional[str] = None

    def __str__(self):
        if self.delimiter is not None:
            return self.delimiter
        elif self.word is not None:
            return self.word
        elif self.number is not None:
            return self.number
        else:
            raise ValueError(
                'Unrecognized type of token, cannot be converted to a string')


def make_tokens_re(word_re: str, grouping: bool) -> str:
    group_type = '' if grouping else '?:'
    return (rf"(?:({group_type}{NUMBER_RE})"
            rf"|({group_type}{word_re})"
            rf"|({group_type}{DELIMITERS_RE}))")


class RegexTokenizer:
    word_chars = rf'[^0-9{PUNCTUATION_RE}\s]'
    word_re = rf"(?:{word_chars}+'[stved]{{1,2}}|{word_chars}+)"
    all_tokens = re.compile(make_tokens_re(word_re, True))
    tokens_as_delims = re.compile(make_tokens_re(word_re, False))

    def apply(self, text: str,
              check_completeness: bool = False) -> Iterable[Token]:
        for match in re.finditer(self.all_tokens, text):
            num, word, delimiter = match.groups()
            if num:
                yield Token(span=match.span(), number=num)
            elif word:
                yield Token(span=match.span(), word=word)
            elif delimiter:
                yield Token(span=match.span(), delimiter=delimiter)
            else:
                raise ValueError("Empty element of the text. "
                                 "Check Tokenizer's regular expressions.")
        if check_completeness:
            undetected = re.sub(
                r'\s+', '',
                ''.join([s for s in re.split(self.tokens_as_delims, text)
                         if len(s) > 0]))
            if undetected:
                raise ValueError(
                    'The text still has some not tokenized parts left')
