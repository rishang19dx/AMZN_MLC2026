import re
import pandas as pd
from unidecode import unidecode
from functools import lru_cache

# Dictionary for legal suffixes
LEGAL_SUFFIXES = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated',
    'llc', 'corp', 'corporation', 'co', 'company', 'llp', 'lp',
    'associates', 'group', 'partners', 'services'
}

# Common address abbreviations mapping (country-agnostic)
ADDRESS_ABBR = {
    'st': 'street',
    'rd': 'road',
    'ave': 'avenue',
    'dr': 'drive',
    'blvd': 'boulevard',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'apt': 'apartment',
    'ste': 'suite',
    'no': 'number'
}

US_STATES = {
    'tx': 'texas',
    'ca': 'california',
    'ny': 'new york',
    'tn': 'tennessee',
    'va': 'virginia',
    'nc': 'north carolina',
    'oh': 'ohio',
    'wa': 'washington',
    'ky': 'kentucky',
    'wi': 'wisconsin'
}

IN_STATES = {
    'mh': 'maharashtra',
    'dl': 'delhi',
    'ka': 'karnataka',
    'up': 'uttar pradesh',
    'gj': 'gujarat',
    'tn': 'tamil nadu' # TN is Tamil Nadu in India
}

@lru_cache(maxsize=100000)
def clean_text(text):
    """
    Lowercases, unidecodes, and removes non-alphanumeric characters (keeps spaces).
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unidecode(text)
    # Replace non-alphanumeric with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_legal_terms(name):
    """
    Extracts and removes common legal terms from the business name.
    Returns (cleaned_name, legal_terms_string).
    """
    if not name:
        return "", ""
    words = name.split()
    base_name_words = []
    legal_terms = []
    
    for w in words:
        if w in LEGAL_SUFFIXES:
            legal_terms.append(w)
        else:
            base_name_words.append(w)
            
    # If the entire name is made of legal terms, keep it as is (rare but possible)
    if not base_name_words:
        return name, ""
        
    return " ".join(base_name_words), " ".join(sorted(list(set(legal_terms))))

def normalize_address(address, country):
    """
    Normalizes common address abbreviations, using country context for states.
    """
    if not address:
        return ""
    words = address.split()
    normalized_words = []
    
    # Select state dictionary based on country
    state_dict = {}
    if country == 'us':
        state_dict = US_STATES
    elif country == 'india':
        state_dict = IN_STATES
        
    for w in words:
        if w in ADDRESS_ABBR:
            normalized_words.append(ADDRESS_ABBR[w])
        elif w in state_dict:
            normalized_words.append(state_dict[w])
        else:
            normalized_words.append(w)
            
    return " ".join(normalized_words)

def preprocess_dataframe(df):
    """
    Applies the full preprocessing pipeline to a dataframe.
    Assumes columns: business_name, business_address, country
    """
    # Normalize country
    df['country_clean'] = df['country'].apply(clean_text)
    
    # Create new columns for the cleaned versions
    df['name_clean'] = df['business_name'].apply(clean_text)
    df['address_clean'] = df['business_address'].apply(clean_text)
    
    # Extract legal terms and clean the name further
    name_splits = df['name_clean'].apply(extract_legal_terms)
    df['name_base'] = name_splits.apply(lambda x: x[0])
    df['name_legal'] = name_splits.apply(lambda x: x[1])
    
    # Normalize address abbreviations with country context
    df['address_norm'] = df.apply(lambda row: normalize_address(row['address_clean'], row['country_clean']), axis=1)
    
    return df

if __name__ == "__main__":
    import config
    import sys
    
    print("Testing preprocessing on Source 2 sample...")
    df_s2 = pd.read_csv(config.TRAIN_S2, sep='\t', nrows=100)
    df_s2_clean = preprocess_dataframe(df_s2)
    
    for _, row in df_s2_clean.head(20).iterrows():
        print(f"Original Name: {row['business_name']}")
        print(f"Base Name:     {row['name_base']}")
        print(f"Legal Terms:   {row['name_legal']}")
        print(f"Original Addr: {row['business_address']}")
        print(f"Norm Addr:     {row['address_norm']}")
        print("-" * 50)
