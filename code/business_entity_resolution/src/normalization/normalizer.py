"""
Normalization module for business entities.

Provides reusable normalizers for:
- Business name (Unicode, legal suffixes, abbreviations, noise cleaning)
- Business address (Road/street abbreviations, unit numbers, numeric tokens)
- Country (Open-set string normalization, never drops unknown countries)
- Blocking key extractors (postal codes, first significant tokens, prefixes)
"""

import re
import unicodedata
from typing import Optional, List, Tuple


# Regex for legal suffixes and corporate forms
# Canonical forms: pvt ltd, ltd, inc, corp, llc, llp, co, gmbh, sa, sas, sarl
LEGAL_SUFFIX_MAP = [
    (r"\b(private\s+limited|pvt\.?\s*ltd\.?|p\.?\s*ltd\.?)\b", " pvt ltd "),
    (r"\b(corporation|corp\.)\b", " corp "),
    (r"\b(incorporated|inc\.)\b", " inc "),
    (r"\b(limited|ltd\.)\b", " ltd "),
    (r"\b(company|co\.)\b", " co "),
    (r"\b(limited\s+liability\s+company|l\.?l\.?c\.?)\b", " llc "),
    (r"\b(limited\s+liability\s+partnership|l\.?l\.?p\.?)\b", " llp "),
    (r"\b(gesellschaft\s+mit\s+beschr[aä]nkter\s+haftung|gmbh)\b", " gmbh "),
    (r"\b(soci[eé]t[eé]\s+anonyme|s\.?a\.?)\b", " sa "),
    (r"\b(soci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e|s\.?a\.?s\.?)\b", " sas "),
    (r"\b(soci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e|s\.?a\.?r\.?l\.?)\b", " sarl "),
    (r"\b(public\s+limited\s+company|plc)\b", " plc "),
]

# Common address token standardizations
ADDRESS_ABBREV_MAP = [
    (r"\b(road|rd\.)\b", " rd "),
    (r"\b(street|st\.)\b", " st "),
    (r"\b(avenue|ave\.)\b", " ave "),
    (r"\b(boulevard|blvd\.)\b", " blvd "),
    (r"\b(lane|ln\.)\b", " ln "),
    (r"\b(drive|dr\.)\b", " dr "),
    (r"\b(highway|hwy\.)\b", " hwy "),
    (r"\b(floor|fl\.)\b", " fl "),
    (r"\b(suite|ste\.)\b", " ste "),
    (r"\b(apartment|apt\.)\b", " apt "),
    (r"\b(building|bldg\.)\b", " bldg "),
    (r"\b(opposite|opp\.)\b", " opp "),
    (r"\b(near|nr\.)\b", " nr "),
    (r"\b(cross|cr\.)\b", " cross "),
    (r"\b(block|blk\.)\b", " blk "),
    (r"\b(nagar|ngr\.)\b", " nagar "),
    (r"\b(marg|mg\.)\b", " marg "),
    (r"\b(sector|sec\.)\b", " sector "),
]

STOP_TOKENS_NAME = {"the", "a", "an", "and", "of", "for", "in", "at", "by", "to"}


def strip_accents(text: str) -> str:
    """Normalize Unicode characters to ASCII representation where possible."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country strings as an OPEN SET.
    Standardizes known representations while preserving any unseen country (e.g. France, Germany).
    Never drops or discards unknown countries.
    """
    if country is None:
        return ""
    c = str(country).strip().upper()
    if not c or c == "NAN":
        return ""
    
    # Common synonyms
    if c in {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA", "U.S.A.", "U.S."}:
        return "US"
    if c in {"INDIA", "IND", "IN", "REPUBLIC OF INDIA"}:
        return "INDIA"
    if c in {"FRANCE", "FR", "FRA", "FRENCH REPUBLIC"}:
        return "FRANCE"
    return c


def normalize_name(name: Optional[str]) -> str:
    """
    Normalize business name:
    - Unicode NFKC + accent flattening
    - Lowercase
    - Replace '&' with 'and'
    - Standardize corporate suffixes (corp, inc, ltd, pvt ltd, etc.)
    - Remove punctuation while retaining alphanumeric and single spaces
    - Strip leading/trailing whitespace
    """
    if name is None:
        return ""
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return ""
    
    s = strip_accents(s)
    s = s.lower()
    
    # Standardize ampersand
    s = re.sub(r"&", " and ", s)
    
    # Standardize legal suffixes
    for pattern, repl in LEGAL_SUFFIX_MAP:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    
    # Remove punctuation except alphanumeric and spaces
    s = re.sub(r"[^\w\s]", " ", s)
    
    # Normalize whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize business address:
    - Unicode NFKC + accent flattening
    - Lowercase
    - Standardize common road/street/unit abbreviations
    - Preserve meaningful numeric tokens (house numbers, PIN/zip codes)
    - Remove punctuation and collapse whitespace
    """
    if address is None:
        return ""
    s = str(address).strip()
    if not s or s.lower() == "nan":
        return ""
    
    s = strip_accents(s)
    s = s.lower()
    
    # Standardize address terms
    for pattern, repl in ADDRESS_ABBREV_MAP:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
        
    # Replace punctuation with space, preserving alphanumeric
    s = re.sub(r"[^\w\s]", " ", s)
    
    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_postal_code(address: Optional[str], country: Optional[str] = "") -> str:
    """
    Extract postal code / PIN code from address:
    - Indian 6-digit PIN code (e.g., 560001, 110001)
    - US 5-digit ZIP code (e.g., 90210, 10001)
    - French 5-digit postal code (e.g., 75001)
    Returns the first matching postal code found, or empty string.
    """
    if not address or str(address).lower() == "nan":
        return ""
    
    s = str(address)
    norm_c = normalize_country(country)
    
    if norm_c == "INDIA":
        # Indian PIN codes: 6 digits starting with 1-9
        m = re.search(r"\b([1-9][0-9]{5})\b", s)
        if m:
            return m.group(1)
            
    if norm_c in {"US", "FRANCE"}:
        # 5-digit postal code
        m = re.search(r"\b([0-9]{5})(?:-[0-9]{4})?\b", s)
        if m:
            return m.group(1)
            
    # Generic fallback: check 6 digits, then 5 digits
    m6 = re.search(r"\b([1-9][0-9]{5})\b", s)
    if m6:
        return m6.group(1)
    m5 = re.search(r"\b([0-9]{5})\b", s)
    if m5:
        return m5.group(1)
        
    return ""


def extract_first_token(normalized_name: str) -> str:
    """
    Extract first significant token of normalized business name,
    skipping common stop words like 'the', 'a', 'an'.
    """
    if not normalized_name:
        return ""
    tokens = normalized_name.split()
    for tok in tokens:
        if tok not in STOP_TOKENS_NAME and len(tok) > 1:
            return tok
    return tokens[0] if tokens else ""


def extract_name_prefix(normalized_name: str, length: int = 4) -> str:
    """Extract first `length` non-whitespace characters from normalized name."""
    if not normalized_name:
        return ""
    cleaned = "".join(normalized_name.split())
    return cleaned[:length] if len(cleaned) >= length else cleaned


def format_record_text(
    name: Optional[str],
    address: Optional[str],
    country: Optional[str],
    use_normalized: bool = True
) -> str:
    """
    Format a record into a structured string representation for neural encoders.
    Retains field structure with clear delimiters:
    name: ... | address: ... | country: ...
    """
    if use_normalized:
        n = normalize_name(name)
        a = normalize_address(address)
        c = normalize_country(country)
    else:
        n = str(name).strip() if name is not None else ""
        a = str(address).strip() if address is not None else ""
        c = str(country).strip() if country is not None else ""
        
    parts = []
    if n:
        parts.append(f"name: {n}")
    if a:
        parts.append(f"address: {a}")
    if c:
        parts.append(f"country: {c}")
        
    return " | ".join(parts) if parts else "empty"
