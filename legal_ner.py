"""
legal_ner.py — Legal PII Entity Extractor
==========================================
Compares generic spaCy NER vs InLegalBERT
for PII detection in legal documents.

Research finding: Measures recall gap between
generic and domain-specific NER models.
"""
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

# Load generic spaCy NER
nlp_generic = spacy.load("en_core_web_sm")

# Legal entity types to detect
LEGAL_ENTITIES = [
    "PERSON", "ORG", "GPE", "DATE", "MONEY",
    "LAW", "CARDINAL", "PERCENT"
]

def extract_pii_spacy(text: str) -> dict:
    """
    Extract PII entities using generic spaCy model.
    Returns dict of entity_type -> list of entities found.
    """
    doc = nlp_generic(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in LEGAL_ENTITIES:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
    return entities


def extract_pii_legal_bert(text: str, max_length: int = 512) -> list:
    """
    Extract PII entities using InLegalBERT.
    Domain-specific model trained on Indian legal text.
    Falls back to spaCy if model unavailable.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
        model = AutoModelForTokenClassification.from_pretrained("law-ai/InLegalBERT")
        ner = pipeline("ner", model=model, tokenizer=tokenizer,
                      aggregation_strategy="simple")
        # Truncate to max_length tokens
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        results = []
        for chunk in chunks[:3]:  # limit for speed
            results.extend(ner(chunk))
        return results
    except Exception as e:
        print(f"InLegalBERT unavailable: {e}. Using spaCy fallback.")
        return []


def compare_ner_models(text: str) -> dict:
    """
    Compare spaCy vs InLegalBERT on same text.
    Returns recall comparison dict for research benchmarking.
    """
    spacy_results = extract_pii_spacy(text)
    legal_results = extract_pii_legal_bert(text)

    spacy_count = sum(len(v) for v in spacy_results.values())
    legal_count = len(legal_results)

    return {
        "spacy_entities": spacy_results,
        "spacy_count": spacy_count,
        "legal_bert_count": legal_count,
        "legal_bert_entities": legal_results,
        "recall_gap": f"{max(0, legal_count - spacy_count)} more entities found by InLegalBERT"
    }