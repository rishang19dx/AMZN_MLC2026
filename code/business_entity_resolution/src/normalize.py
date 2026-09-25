"""
Country-agnostic text normalisation shared by blocking and matching.

Deliberately rule-light: no hand-made legal-suffix or state lists (those would
be US/India-specific and fail on France). Frequent tokens such as "pvt",
"limited", "sarl" are down-weighted downstream by IDF, learned from the data.

Transliteration:
  * Indic-script words are looked up in a dictionary learned from labelled
    train pairs (src/translit.py -> $BER_CACHE_DIR/translit.json); see
    docs/FINDINGS.md §3.1 (native-script name similarity 67 -> 98).
  * Everything else, and dictionary misses, goes through anyascii (ISC
    licence; replaces Unidecode, which is GPL-2 and transliterates Indic
    scripts worse: 73.5 vs 67.0 similarity in FINDINGS).
If translit.json does not exist, only anyascii is used.
"""

import json
import os
import re
import string

from anyascii import anyascii

_NON_ALNUM = re.compile(r'[^a-z0-9]+')
# ASCII punctuation plus Indic danda marks. Deliberately not \W: Indic
# combining signs (virama, vowel signs) are not "word" characters to Python
# and would be stripped from the end of words.
PUNCT = string.punctuation + '।॥'
_DICT_PATH = os.path.join(os.environ.get('BER_CACHE_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'cache')),
    'translit.json')
_dict = None


def _translit_dict():
    global _dict
    if _dict is None:
        try:
            with open(_DICT_PATH, encoding='utf-8') as f:
                _dict = json.load(f)['words']
        except FileNotFoundError:
            _dict = {}
    return _dict


def has_indic(text):
    """True if the text contains a character from the Indic script blocks (U+0900-U+0DFF)."""
    return any('ऀ' <= ch <= '෿' for ch in text or '')


def is_non_latin(text):
    """True if the text contains characters outside Latin / Latin-extended."""
    return any(ord(ch) > 0x24F for ch in text or '')


def norm_latin(text):
    """anyascii -> lowercase -> '&' to 'and' -> every non-alphanumeric run to one space."""
    if not text:
        return ''
    t = (text if text.isascii() else anyascii(text)).lower().replace('&', ' and ')
    return _NON_ALNUM.sub(' ', t).strip()


def norm(text):
    """norm_latin, but Indic-script words are first mapped with the learned dictionary."""
    if not text:
        return ''
    if has_indic(text):
        d = _translit_dict()
        text = ' '.join(d.get(w.strip(PUNCT)) or w for w in text.split())
    return norm_latin(text)
