"""
argument_miner.py — Legal Argument Structure Mining
=====================================================
Classifies legal text into argument components:
- Premise: factual statements supporting a claim
- Claim: main legal assertion
- Exception: carve-outs and exclusions
- Obligation: duties and requirements

Extends marker-based approach from Anand et al. 2025.
Visualizes clause dependency graph using NetworkX.
"""
from transformers import pipeline
import networkx as nx
from typing import List, Dict

ARGUMENT_LABELS = ["Premise", "Claim", "Exception", "Obligation", "Definition"]


def classify_argument_component(text: str) -> dict:
    """
    Classify text into legal argument structure component.
    """
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small"
        )
        result = classifier(text, ARGUMENT_LABELS)
        return {
            "component": result["labels"][0],
            "confidence": round(result["scores"][0], 3),
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        }
    except Exception as e:
        return {"component": "Unknown", "confidence": 0.0, "error": str(e)}


def mine_arguments(clauses: List[str]) -> List[Dict]:
    """
    Mine argument structure from list of clauses.
    Returns structured argument components.
    """
    results = []
    for i, clause in enumerate(clauses):
        result = classify_argument_component(clause)
        result["clause_id"] = i
        results.append(result)
    return results


def build_argument_graph(mined_args: List[Dict]) -> nx.DiGraph:
    """
    Build directed graph showing clause dependencies.
    Premises point to Claims they support.
    Exceptions point to Claims they modify.
    """
    G = nx.DiGraph()

    claims = [a for a in mined_args if a["component"] == "Claim"]
    premises = [a for a in mined_args if a["component"] == "Premise"]
    exceptions = [a for a in mined_args if a["component"] == "Exception"]
    obligations = [a for a in mined_args if a["component"] == "Obligation"]

    # Add nodes
    for arg in mined_args:
        G.add_node(
            arg["clause_id"],
            label=arg["component"],
            text=arg.get("text_preview", ""),
            confidence=arg.get("confidence", 0)
        )

    # Add edges: premises → nearest claim
    for p in premises:
        for c in claims:
            if abs(p["clause_id"] - c["clause_id"]) <= 2:
                G.add_edge(p["clause_id"], c["clause_id"],
                          relation="supports")

    # Exceptions → claims they modify
    for e in exceptions:
        for c in claims:
            if abs(e["clause_id"] - c["clause_id"]) <= 3:
                G.add_edge(e["clause_id"], c["clause_id"],
                          relation="modifies")

    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """
    Summarize argument graph structure.
    """
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "claims": len([n for n, d in G.nodes(data=True) if d.get("label") == "Claim"]),
        "premises": len([n for n, d in G.nodes(data=True) if d.get("label") == "Premise"]),
        "exceptions": len([n for n, d in G.nodes(data=True) if d.get("label") == "Exception"]),
    }