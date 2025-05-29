import spacy
from typing import Dict, List
import logging

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    raise

def extract_entities(text: str) -> List[Dict[str, List[str]]]:
    """
    Extract ORG and GPE entities from text using spaCy.
    
    Args:
        text: Text extracted from PDF
        
    Returns:
        List of dictionaries containing ORG and GPE entities.
        Example:
        [
            {
                'organizations': ['United Nations', 'World Bank'],
                'geopolitical_entities': ['New York', 'France']
            },
            {
                'organizations': ['Google'],
                'geopolitical_entities': ['California']
            }
        ]
        
    """
    # Split text into pages (assuming pages are separated by double newlines)
    pages = text.split("\n\n")
    
    results = []
    
    for page_text in pages:
        if not page_text.strip():
            continue
            
        # Process the text with spaCy
        doc = nlp(page_text)
        
        # Extract ORG and GPE entities
        orgs = []
        gpes = []
        
        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.text not in orgs:
                orgs.append(ent.text)
            elif ent.label_ == "GPE" and ent.text not in gpes:
                gpes.append(ent.text)
        
        # Add results for this page
        results.append({
            "organizations": orgs,
            "geopolitical_entities": gpes
        })
    
    return results

def extract_entities_from_folder(
    documents_dict: Dict[str, str]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract entities from documents in a folder.
    
    Args:
        documents_dict: Dictionary containing document names and their text
        
    Returns:
        Dictionary with folder names as keys and lists of entity dictionaries as values.
    """
    extracted: Dict[str, Dict[str, List[str]]] = {}

    for fname, text in documents_dict.items():
        pages = extract_entities(text)
        all_orgs, all_gpes = [], []
        for pg in pages:
            all_orgs += pg.get("organizations", [])
            all_gpes += pg.get("geopolitical_entities", [])
        extracted[fname] = {
            "organizations": list(dict.fromkeys(all_orgs)),
            "geopolitical_entities": list(dict.fromkeys(all_gpes))
        }

    return extracted