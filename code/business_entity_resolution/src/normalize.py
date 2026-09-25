"""
Country-agnostic text normalisation shared by blocking and matching.

Deliberately rule-light: no hand-made legal-suffix or state lists (those would
be US/India-specific and fail on France). Frequent tokens such as "pvt",
"limited", "sarl" are down-weighted downstream by IDF, learned from the data.
"""

import re

from unidecode import unidecode

_NON_ALNUM = re.compile(r'[^a-z0-9]+')


def norm(text):
    """Transliterate to ASCII (handles Indic scripts, accents), lowercase,
    '&' -> 'and', every non-alphanumeric run -> one space."""
    if not text:
        return ''
    t = unidecode(text).lower().replace('&', ' and ')
    return _NON_ALNUM.sub(' ', t).strip()


def is_non_latin(text):
    """True if the text contains characters outside Latin / Latin-extended
    (e.g. Kannada, Malayalam, Devanagari)."""
    return any(ord(ch) > 0x24F for ch in text or '')
