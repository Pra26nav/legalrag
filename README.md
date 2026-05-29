# ⚖️ LegalRAG — Privacy-Aware Legal Document Intelligence

Research system combining RAG, NLP, and multi-component legal analysis.

## Research Contributions

| Component | Research Value |
|---|---|
| PII Detection | Measures recall gap: generic spaCy vs InLegalBERT |
| Clause Classification | Zero-shot CUAD clause typing using DeBERTa |
| Argument Mining | Premise/Claim/Exception structure (extends Anand et al. 2025) |
| Privacy Mode | Cloud vs local inference latency/accuracy trade-off |

## Architecture
Legal PDF Upload
|
PyPDF Text Extraction + MD5 Caching
|
FAISS Vector Index (LangChain)
|
4 Analysis Tabs:
Tab 1: RAG Q&A (Groq/Ollama toggle)
Tab 2: PII Detection (spaCy vs InLegalBERT)
Tab 3: Clause Classification (DeBERTa CUAD)
Tab 4: Argument Mining (NetworkX graph)

## Tech Stack
- LangChain + FAISS — RAG pipeline
- HuggingFace Transformers — InLegalBERT, DeBERTa
- spaCy — baseline NER
- Groq (Llama 3.3 70B) — cloud inference
- Ollama — privacy-preserving local inference
- NetworkX — argument graph visualization
- Streamlit — research UI

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Add `.env`:
GROQ_API_KEY=your_key_here

Run:
```bash
streamlit run app.py
```

## Research Benchmarks
- CUAD dataset: https://huggingface.co/datasets/cuad
- InLegalBERT: https://huggingface.co/law-ai/InLegalBERT
- Baseline: generic spaCy en_core_web_sm