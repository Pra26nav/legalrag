"""
privacy_mode.py — Privacy-Preserving Inference Toggle
======================================================
Switches between:
- Cloud mode: Groq API (fast, accurate)
- Privacy mode: Ollama local (private, slower)

Research finding: Measures accuracy/latency trade-off
between cloud and local inference for legal documents.
"""
import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def get_cloud_llm():
    """Groq cloud LLM — fast, accurate, requires API key."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def get_local_llm():
    """
    Ollama local LLM — private, no data leaves device.
    Requires Ollama installed: https://ollama.ai
    Run: ollama pull llama3.2
    """
    try:
        from langchain_community.llms import Ollama
        return Ollama(model="llama3.2", temperature=0.1)
    except Exception as e:
        raise RuntimeError(f"Ollama not available: {e}. Install from https://ollama.ai")


def query_with_benchmark(question: str, context: str, mode: str = "cloud") -> dict:
    """
    Query LLM and measure latency for benchmarking.

    Args:
        question: User question
        context: Document context
        mode: 'cloud' (Groq) or 'privacy' (Ollama)

    Returns:
        dict with answer, latency, mode used
    """
    start = time.time()

    system_prompt = f"""You are a legal document analyst.
Answer based ONLY on the provided legal document context.
If not found, say "Not found in document."
Be precise and cite specific clauses.

DOCUMENT CONTEXT:
{context}"""

    try:
        if mode == "privacy":
            llm = get_local_llm()
            messages = f"{system_prompt}\n\nQuestion: {question}"
            answer = llm.invoke(messages)
        else:
            llm = get_cloud_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            response = llm.invoke(messages)
            answer = response.content

        latency = round(time.time() - start, 2)
        return {
            "answer": answer,
            "latency_seconds": latency,
            "mode": mode,
            "model": "llama-3.3-70b (Groq)" if mode == "cloud" else "llama3.2 (Ollama local)"
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "latency_seconds": -1,
            "mode": mode,
            "error": True
        }