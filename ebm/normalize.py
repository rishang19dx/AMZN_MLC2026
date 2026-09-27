"""
Record normalisation and tokenisation for the pair energy model (v2).

Every rule below targets a noise pattern measured on the training data
(see the main repo's docs/FINDINGS.md). Source 1 is perfectly clean and all
noise is on the S2/S3 side, so rules map noisy variants to Source 1's form.

    text -> NFKC -> native-script words via a dictionary learned from train
         -> accents stripped -> lowercase -> dotted initialisms collapsed
         -> legal forms canonicalised (one pass) -> honorifics dropped
         -> leetspeak fixed (names) -> web names reduced to their core
         -> address abbreviations / ordinals / states canonicalised (one pass)
         -> numbers split out (house, unit, postcode, others)
         -> repeated words removed

Tokens are field-tagged and hashed (crc32, stable across runs and processes)
into HASH_BUCKETS ids; each token also carries a field id for the model.
Country is kept as a plain string token: France (unseen in train) needs no
special code, and nothing is keyed on a country list.
"""

import re
import unicodedata
import zlib

HASH_BUCKETS = 1 << 20
SEQ_LEN = 96

# field ids (0 = padding)
F_NAME, F_NAME3, F_LEGAL, F_ADDR, F_NUM, F_HOUSE, F_UNIT, F_ZIP, F_COUNTRY = range(1, 10)
N_FIELDS = 10
BUDGET = {F_NAME: 16, F_NAME3: 28, F_LEGAL: 4, F_ADDR: 24, F_NUM: 8, F_HOUSE: 1, F_UNIT: 2, F_ZIP: 1, F_COUNTRY: 1}

# ---------------------------------------------------------------------------
# lookup tables
# ---------------------------------------------------------------------------

LEGAL = {
    # India / UK
    'private limited': 'pvt ltd', 'pvt limited': 'pvt ltd', 'private ltd': 'pvt ltd', 'pvt ltd': 'pvt ltd',
    'p ltd': 'pvt ltd', 'pvt': 'pvt', 'private': 'pvt', 'limited': 'ltd', 'ltd': 'ltd', 'llp': 'llp',
    # US
    'incorporated': 'inc', 'inc': 'inc', 'corporation': 'corp', 'corp': 'corp', 'company': 'co', 'co': 'co',
    'llc': 'llc', 'l l c': 'llc', 'lp': 'lp', 'pllc': 'pllc', 'pc': 'pc', 'pa': 'pa', 'plc': 'plc',
    # France
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sa': 'sa', 'sci': 'sci', 'snc': 'snc',
    'scop': 'scop', 'gie': 'gie', 'societe': 'societe', 'cie': 'cie', 'et cie': 'cie',
}
HONORIFICS = {'m s', 'ms', 'messrs', 'shri', 'sri', 'smt', 'mr', 'mrs', 'dr', 'the', 'm/s'}

ADDR_ABBR = {
    # US
    'st': 'street', 'str': 'street', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'pkwy': 'parkway', 'hwy': 'highway',
    'sq': 'square', 'ter': 'terrace', 'cir': 'circle', 'trl': 'trail', 'expy': 'expressway', 'fwy': 'freeway',
    'mt': 'mount', 'ft': 'fort', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    'apt': 'apartment', 'ste': 'suite', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building', 'rm': 'room',
    # India
    'nr': 'near', 'opp': 'opposite', 'sec': 'sector', 'sect': 'sector', 'mkt': 'market', 'marg': 'marg',
    'ngr': 'nagar', 'clny': 'colony', 'extn': 'extension', 'dist': 'district', 'distt': 'district',
    'tq': 'taluk', 'tal': 'taluk', 'po': 'post', 'ps': 'police station', 'hno': 'house', 'h no': 'house',
    # France
    'r': 'rue', 'bd': 'boulevard', 'bld': 'boulevard', 'fg': 'faubourg', 'fbg': 'faubourg', 'chem': 'chemin',
    'imp': 'impasse', 'all': 'allee', 'rte': 'route', 'qu': 'quai', 'sq': 'square', 'res': 'residence',
    'cedex': '', 'bis': 'bis', 'ter': 'ter', 'no': '', 'nos': '', 'number': '', 'plot': 'plot',
}
ORDINALS = {w: str(i) for i, w in enumerate(
    'zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth '
    'fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split())}
US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'colorado': 'co',
    'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la',
    'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx',
    'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc',
}
STATE_TOKEN = {v: 'state_' + v for v in US_STATES.values()}   # 2-letter codes only as the last address words


def _alternation(keys):
    keys = sorted(keys, key=len, reverse=True)       # longest first: "private limited" before "private"
    return re.compile(r'\b(' + '|'.join(re.escape(k) for k in keys) + r')\b')


_LEGAL_RE = _alternation(LEGAL)
_STATE_RE = _alternation(US_STATES)
_DOTTED_RE = re.compile(r'\b(?:[a-z]\.){2,}[a-z]?\.?')   # l.l.c. -> llc, p.v.t -> pvt
_NONWORD_RE = re.compile(r'[^\w]+')
_WEB_RE = re.compile(r'\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]{2,})\.(?:com|in|net|org|fr|co|biz|info|us)\b')
_MS_RE = re.compile(r'^\s*(?:m\s*/\s*s|messrs)\.?\s+')
_LEET = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '5': 's', '@': 'a', '$': 's'})
_DIGIT_RE = re.compile(r'\d+')
_NUMSUFFIX_RE = re.compile(r'^(\d+)(?:st|nd|rd|th|[a-z])?$')
_INDIC = re.compile('[ऀ-෿]')


def _strip_accents(s):
    if s.isascii():
        return s
    out = []
    for ch in unicodedata.normalize('NFKD', s):
        if not unicodedata.combining(ch):
            out.append(ch)
    return ''.join(out)


def _translit(s, dictionary):
    if not dictionary or not _INDIC.search(s):
        return s
    return ' '.join(dictionary.get(w, w) for w in s.split())


def _base(s, dictionary):
    s = unicodedata.normalize('NFKC', s or '')
    s = _translit(s, dictionary)
    s = _strip_accents(s).lower()
    s = s.replace('&', ' and ').replace('@', ' at ')
    s = _DOTTED_RE.sub(lambda m: m.group(0).replace('.', ''), s)
    return s


def _dedupe(words):
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    return out


def _leet(w):
    # only inside mostly-alphabetic words: "onc0logy" -> "oncology", "lnvestment" is left alone
    if w.isalpha() or w.isdigit() or sum(c.isalpha() for c in w) < 2:
        return w
    return w.translate(_LEET)


def normalize_name(name, dictionary=None):
    """-> (core words, legal forms). Core words drive matching; legal forms are a weak signal."""
    s = _base(name, dictionary)
    s = _MS_RE.sub(' ', s)
    s = _WEB_RE.sub(lambda m: ' ' + m.group(1).replace('-', ' ') + ' ', s)
    s = _NONWORD_RE.sub(' ', s)
    legal = []

    def keep_legal(m):
        legal.append(LEGAL[m.group(1)])
        return ' '
    s = _LEGAL_RE.sub(keep_legal, s)
    words = []
    for w in s.split():
        if w in HONORIFICS:
            continue
        words.append(_leet(w))
    return _dedupe(words), sorted(set(legal))


def normalize_address(addr, dictionary=None):
    """-> (words, house number, units, postcode, other numbers)."""
    s = _base(addr, dictionary)
    parts = [p.strip() for p in s.split(',')]
    for i in {0, len(parts) - 1}:
        p = parts[i].rstrip('.')
        if p in US_STATES:
            parts[i] = 'state_' + US_STATES[p]
        elif p in STATE_TOKEN and len(parts) > 1:
            parts[i] = STATE_TOKEN[p]
    s = ', '.join(parts)
    unit = re.findall(r'(?:unit|suite|ste|apt|apartment|flat|room|rm|#)\s*[-:#]?\s*([a-z]?\d+[a-z]?)\b', s)
    s = _NONWORD_RE.sub(' ', s)
    words, nums = [], []
    for w in s.split():
        w = ORDINALS.get(w, w)
        m = _NUMSUFFIX_RE.match(w)
        if m:                                   # 009291 -> 9291, 11th -> 11, 470c -> 470
            nums.append(m.group(1).lstrip('0') or '0')
            continue
        if any(c.isdigit() for c in w):         # mixed tokens like "g109", "b3"
            nums.extend(d.lstrip('0') or '0' for d in _DIGIT_RE.findall(w))
            w = re.sub(r'\d+', '', w)
            if len(w) < 2:
                continue
        w = ADDR_ABBR.get(w, w)
        if w:
            words.extend(w.split())
    zips = [n for n in nums if len(n) in (5, 6)]
    house = next((n for n in nums if 0 < len(n) <= 5 and n not in zips), '')
    units = [u.lstrip('0') for u in unit][:BUDGET[F_UNIT]]
    others = [n for n in nums if n != house and n not in zips and n not in units]
    return _dedupe(words), house, units, zips[:1], others


def _h(field, tok):
    return zlib.crc32(f'{field}:{tok}'.encode('utf-8', 'ignore')) % (HASH_BUCKETS - 1) + 1


def tokenize(name, addr, country, dictionary=None):
    """-> (token ids, field ids), each at most SEQ_LEN long, in a fixed field order."""
    core, legal = normalize_name(name, dictionary)
    words, house, units, zips, others = normalize_address(addr, dictionary)
    toks, fields = [], []

    def add(field, items):
        for t in items[:BUDGET[field]]:
            toks.append(_h(field, t))
            fields.append(field)
    add(F_NAME, core)
    compact = ' '.join(core)
    add(F_NAME3, [compact[i:i + 3] for i in range(max(0, len(compact) - 2))])
    add(F_LEGAL, legal)
    add(F_HOUSE, [house] if house else [])
    add(F_UNIT, units)
    add(F_ZIP, zips)
    add(F_ADDR, words)
    add(F_NUM, others)
    add(F_COUNTRY, [(country or '').strip().lower()])
    return toks[:SEQ_LEN], fields[:SEQ_LEN]


def key_tokens(toks, fields):
    """Tokens used to mine look-alike negatives: name words and the house number."""
    return [t for t, f in zip(toks, fields) if f in (F_NAME, F_HOUSE)]
