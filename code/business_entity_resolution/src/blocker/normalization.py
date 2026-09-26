"""
Reusable field normalisation (blocking now, the final matcher later).

Built on src/normalize.py (anyascii transliteration, learned Indic-script word
dictionary, lowercasing, '&' -> 'and', punctuation -> spaces) plus token rules:

  * acronym runs:    "s a s" -> "sas", "c i t colony" -> "cit colony"
  * ordinals:        "2nd" -> "2", "45th" / "45nd" (typo) -> "45"; leading zeros dropped
  * null tokens:     "null", "none", "nan", "nil" removed (targets contain them)
  * names:           legal-form spellings unified ("pvt" -> "private", "ltd" -> "limited", ...)
  * name_core:       the name without legal forms / "the" / "and" (for keys and char n-grams)
  * addresses:       street-type abbreviations unified, English and French
                     ("rd" -> "road", "r" -> "rue", "bd" -> "boulevard", ...)
  * postal codes:    5- or 6-digit tokens of the raw address
  * country:         open-set string; only spelling variants of the same label are unified

Raw fields are never modified: prepare_records() adds new columns next to them.
The rule tables are generic (no business data); nothing is keyed on the set of
training countries, so an unseen country passes straight through.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

import normalize as base
from translit import learn_table

NULL_TOKENS = frozenset({'null', 'none', 'nan', 'nil'})

NAME_ABBREV = {
    'pvt': 'private', 'pvte': 'private', 'priv': 'private', 'ltd': 'limited', 'ltda': 'limited',
    'lmt': 'limited', 'co': 'company', 'cos': 'companies', 'corp': 'corporation', 'inc': 'incorporated',
    'incorp': 'incorporated', 'intl': 'international', 'mfg': 'manufacturing',
    'bros': 'brothers', 'svc': 'services', 'svcs': 'services', 'mgmt': 'management',
    'assn': 'association', 'assoc': 'association', 'ent': 'enterprises', 'ents': 'enterprises',
    'grp': 'group', 'hldgs': 'holdings', 'natl': 'national', 'dept': 'department',
    'univ': 'university', 'hosp': 'hospital', 'med': 'medical', 'ctr': 'center', 'centre': 'center',
    'et': 'and', 'n': 'and',
}
LEGAL_FORMS = frozenset({
    'private', 'limited', 'company', 'companies', 'corporation', 'incorporated', 'llc', 'llp', 'lp', 'plc',
    'pllc', 'pc', 'opc', 'public', 'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'snc', 'gmbh', 'ag', 'bv',
    'the', 'and', 'of', 'de', 'du', 'des', 'la', 'le', 'les',
})
ADDR_ABBREV = {
    'st': 'street', 'str': 'street', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue', 'avn': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'bvd': 'boulevard', 'dr': 'drive', 'drv': 'drive',
    'ln': 'lane', 'ct': 'court', 'crt': 'court', 'cir': 'circle', 'pkwy': 'parkway', 'hwy': 'highway',
    'ter': 'terrace', 'terr': 'terrace', 'pl': 'place', 'sq': 'square', 'ste': 'suite', 'apt': 'apartment',
    'fl': 'floor', 'flr': 'floor', 'bldg': 'building', 'opp': 'opposite', 'nr': 'near', 'ngr': 'nagar',
    'mkt': 'market', 'stn': 'station', 'hosp': 'hospital', 'r': 'rue', 'imp': 'impasse', 'ch': 'chemin',
    'che': 'chemin', 'rte': 'route', 'all': 'allee', 'crs': 'cross', 'extn': 'extension', 'ext': 'extension',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west', 'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest', 'mt': 'mount', 'ft': 'fort', 'jn': 'junction', 'jct': 'junction',
}
COUNTRY_ALIASES = {
    'us': 'us', 'usa': 'us', 'u s': 'us', 'u s a': 'us', 'united states': 'us',
    'united states of america': 'us', 'america': 'us',
    'india': 'india', 'in': 'india', 'ind': 'india', 'bharat': 'india',
    'france': 'france', 'fr': 'france', 'fra': 'france', 'republique francaise': 'france',
}

_ORDINAL = re.compile(r'^(\d+)(st|nd|rd|th)$')
_POSTAL = re.compile(r'(?<![\d])(\d{6}|\d{5})(?![\d])')
_DIGIT = re.compile(r'\d')
_SPACE = re.compile(r'\s+')


# ---------------------------------------------------------------------------
# Learned transliteration dictionary (training pairs only)
# ---------------------------------------------------------------------------

def set_translit_dict(words):
    """Use `words` (native word -> Latin word) for every later normalisation call
    in this process (and in worker processes forked after this call)."""
    base._dict = dict(words or {})


def learn_translit_dict(data, min_count=3, min_share=0.5):
    """Learn the dictionary from the true pairs of a (training) ERData."""
    i, j = data.positive_pairs()
    s1n, s1a = data.s1['business_name'].to_numpy(), data.s1['business_address'].to_numpy()
    tgn, tga = data.tg['business_name'].to_numpy(), data.tg['business_address'].to_numpy()
    native = np.fromiter((base.has_indic(n) or base.has_indic(a) for n, a in zip(tgn[j], tga[j])), bool, len(j))
    pairs = (((s1n[a], s1a[a]), (tgn[b], tga[b])) for a, b in zip(i[native], j[native]))
    return learn_table(pairs, min_count, min_share)


# ---------------------------------------------------------------------------
# Token rules
# ---------------------------------------------------------------------------

def _collapse_acronyms(tokens):
    """Join runs of >= 2 single-letter tokens: ['s', 'a', 's'] -> ['sas']."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.extend([''.join(run)] if len(run) > 1 else run)
            run = []
        out.append(t)
    if run:
        out.extend([''.join(run)] if len(run) > 1 else run)
    return out


def _canon_number(t):
    m = _ORDINAL.match(t)
    if m:
        t = m.group(1)
    if t.isdigit():
        t = t.lstrip('0') or '0'
    return t


def _tokens(text, collapse=True):
    toks = [t for t in base.norm(text).split() if t not in NULL_TOKENS]
    if collapse:
        toks = _collapse_acronyms(toks)
    return [_canon_number(t) for t in toks]


def normalize_name(text, expand=True, collapse=True):
    toks = _tokens(text, collapse)
    if expand:
        toks = [NAME_ABBREV.get(t, t) for t in toks]
    return ' '.join(toks)


def name_core(name_n):
    """Distinctive part of a normalised name (legal forms and function words removed)."""
    core = [t for t in name_n.split() if t not in LEGAL_FORMS]
    return ' '.join(core) if core else name_n


def normalize_address(text, expand=True, collapse=True):
    toks = _tokens(text, collapse)
    if expand:
        toks = [ADDR_ABBREV.get(t, t) for t in toks]
    return ' '.join(toks)


def normalize_country(text):
    c = base.norm_latin(text or '')
    return COUNTRY_ALIASES.get(c, c) or 'unknown'


def extract_postal(raw_address):
    """5/6-digit tokens of the raw address (Indian PIN, US ZIP, French code postal)."""
    return _POSTAL.findall(raw_address or '')


def house_number(addr_n):
    for t in addr_n.split():
        if _DIGIT.search(t):
            return t
    return ''


def raw_view(text):
    """Light normalisation that keeps the original script (Pipeline B input):
    Unicode NFKC, case folding, whitespace collapse, 'null' removed."""
    t = unicodedata.normalize('NFKC', text or '').casefold()
    return ' '.join(w for w in _SPACE.split(t) if w and w.strip('.,;:-') not in NULL_TOKENS)


def blocking_keys(name_core_s, addr_n, postal):
    """Deterministic blocking keys (Pipeline C 'keys' pass). Several keys per
    record; retrieval takes the union weighted by key rarity, so no single key
    has to be exact."""
    keys = []
    nt = [t for t in name_core_s.split() if len(t) > 1]
    if nt:
        keys.append('f:' + nt[0])                              # first significant name token
        keys.append('p:' + ''.join(nt)[:4])                    # 4-char prefix, spacing-insensitive
        if len(nt) > 1:
            keys.append('s:' + '_'.join(sorted(nt[:2])))       # word-order-insensitive pair
            keys.append('l:' + nt[-1])                         # last significant token
    at = addr_n.split()
    for k, t in enumerate(at):
        if _DIGIT.search(t):
            nxt = next((u for u in at[k + 1:k + 3] if u.isalpha() and len(u) > 2), '')
            if nxt:
                keys.append(f'h:{t}_{nxt}')                    # house number + street word
            break
    alpha = [t for t in at if t.isalpha() and len(t) > 3]
    if len(alpha) >= 2:
        keys.append('a:' + '_'.join(sorted(alpha[:2])))
    keys.extend('z:' + z for z in postal)
    return ' '.join(keys)


# ---------------------------------------------------------------------------
# DataFrame preparation
# ---------------------------------------------------------------------------

def _map_unique(series, fn):
    codes, uniques = pd.factorize(series, sort=False)
    mapped = np.array([fn(u) for u in uniques], dtype=object)
    return pd.Series(mapped[codes] if len(uniques) else np.array([], dtype=object), index=series.index)


def _prepare_chunk(args):
    df, expand, collapse = args
    out = pd.DataFrame(index=df.index)
    out['name_n'] = _map_unique(df['business_name'], lambda s: normalize_name(s, expand, collapse))
    out['addr_n'] = _map_unique(df['business_address'], lambda s: normalize_address(s, expand, collapse))
    out['country_n'] = _map_unique(df['country'], normalize_country)
    out['name_core'] = _map_unique(out['name_n'], name_core)
    postal = df['business_address'].map(extract_postal)
    out['keys'] = [blocking_keys(c, a, z) for c, a, z in zip(out['name_core'], out['addr_n'], postal)]
    return out


def prepare_records(df, cfg=None, workers=1):
    """Add normalised columns (name_n, addr_n, country_n, name_core, full_n, keys)
    to a raw record frame, in place, and return it."""
    ncfg = (cfg or {}).get('normalization', {})
    expand = ncfg.get('expand_abbreviations', True)
    collapse = ncfg.get('collapse_acronyms', True)
    cols = ['business_name', 'business_address', 'country']
    if workers > 1 and len(df) > 200_000:
        import multiprocessing as mp
        chunks = [(df[cols].iloc[s:s + 250_000], expand, collapse) for s in range(0, len(df), 250_000)]
        with mp.get_context('fork').Pool(workers) as pool:
            out = pd.concat(pool.map(_prepare_chunk, chunks))
    else:
        out = _prepare_chunk((df[cols], expand, collapse))
    for c in out.columns:
        df[c] = out[c].to_numpy()
    df['full_n'] = df['name_n'] + ' ' + df['addr_n']
    return df


def prepare_data(data, cfg=None, workers=1):
    prepare_records(data.s1, cfg, workers)
    prepare_records(data.tg, cfg, workers)
    return data
