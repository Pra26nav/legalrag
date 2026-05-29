"""
clause_classifier.py — Legal Clause Type Classifier
=====================================================
Classifies legal clauses into CUAD-compatible types.
Uses zero-shot classification as baseline.
Can be fine-tuned on CUAD 41-class dataset.

CUAD dataset: https://huggingface.co/datasets/cuad
"""
from transformers import pipeline

# 41 CUAD clause types (condensed to top 15 for demo)
CLAUSE_TYPES = [
    "Limitation of Liability",
    "Indemnification",
    "Termination for Convenience",
    "Governing Law",
    "Dispute Resolution",
    "Confidentiality",
    "Intellectual Property",
    "Non-Compete",
    "Warranty",
    "Payment Terms",
    "Force Majeure",
    "Assignment",
    "Audit Rights",
    "Insurance",
    "Change of Control"
]


def classify_clause(text: str, threshold: float = 0.3) -> dict:
    """
    Classify a legal clause into CUAD clause types.
    Uses zero-shot classification with DeBERTa.

    Args:
        text: Legal clause text
        threshold: Minimum confidence to report

    Returns:
        dict with top clause type and confidence scores
    """
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small"
        )
        result = classifier(text, CLAUSE_TYPES, multi_label=True)

        # Filter by threshold
        filtered = [
            {"label": l, "score": round(s, 3)}
            for l, s in zip(result["labels"], result["scores"])
            if s >= threshold
        ]

        return {
            "top_clause_type": result["labels"][0],
            "confidence": round(result["scores"][0], 3),
            "all_matches": filtered[:5]
        }
    except Exception as e:
        return {"error": str(e), "top_clause_type": "Unknown"}


def batch_classify(clauses: list) -> list:
    """
    Classify multiple clauses at once.
    Returns list of classification results.
    """
    return [classify_clause(clause) for clause in clauses]