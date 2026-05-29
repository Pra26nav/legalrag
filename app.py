"""
app.py — LegalRAG: Privacy-Aware Legal Document Intelligence
=============================================================
Research system combining:
- Legal PII detection (spaCy vs InLegalBERT comparison)
- CUAD clause type classification (DeBERTa zero-shot)
- Argument structure mining (Premise/Claim/Exception/Obligation)
- Privacy mode toggle (Groq cloud vs Ollama local)
- RAG Q&A grounded in uploaded legal documents

Stack: LangChain + FAISS + Groq/Ollama + Streamlit + Transformers
"""
import streamlit as st
import os
import hashlib
import pickle
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from legal_ner import compare_ner_models, extract_pii_spacy
from clause_classifier import classify_clause, batch_classify
from argument_miner import mine_arguments, build_argument_graph, get_graph_summary
from privacy_mode import query_with_benchmark

load_dotenv()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

st.set_page_config(
    page_title="LegalRAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state
for key in ["vector_store", "chat_history", "doc_meta", "raw_text", "chunks"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []

# Hero
st.markdown("""
<div class="hero">
    <h1>⚖️ LegalRAG</h1>
    <p>Privacy-Aware Legal Document Intelligence — PII Detection · Clause Classification · Argument Mining</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📄 Upload Contract")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Legal PDF", type=["pdf"])
    privacy_mode = st.radio("Inference Mode", ["☁️ Cloud (Groq)", "🔒 Privacy (Ollama local)"], index=0)
    mode = "privacy" if "Privacy" in privacy_mode else "cloud"

    if uploaded_file:
        st.markdown(f"**✓ {uploaded_file.name}**")
        chunk_size = st.slider("Chunk Size", 200, 800, 400, 100)

        if st.button("⚖️ Process Document"):
            os.makedirs("cache", exist_ok=True)
            cache_key = hashlib.md5((uploaded_file.name + str(uploaded_file.size)).encode()).hexdigest()
            cache_path = f"cache/{cache_key}.faiss"
            meta_path = f"cache/{cache_key}.meta"

            with st.spinner("Processing legal document..."):
                if os.path.exists(cache_path):
                    embeddings = get_embeddings()
                    vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                    st.info("⚡ Loaded from cache!")
                else:
                    progress = st.progress(0)
                    reader = PdfReader(uploaded_file)
                    all_text = ""
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            all_text += f"\n[PAGE {i+1} | {uploaded_file.name}]\n{text}"
                        progress.progress((i+1) / len(reader.pages) * 0.4)

                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
                    chunks = splitter.split_text(all_text)
                    progress.progress(0.6)

                    embeddings = get_embeddings()
                    vector_store = FAISS.from_texts(chunks, embeddings)
                    progress.progress(0.9)

                    vector_store.save_local(cache_path)
                    meta = {"file": uploaded_file.name, "pages": len(reader.pages), "chunks": len(chunks)}
                    with open(meta_path, "wb") as f:
                        pickle.dump(meta, f)

                    st.session_state.raw_text = all_text
                    st.session_state.chunks = chunks
                    progress.progress(1.0)

                st.session_state.vector_store = vector_store
                st.session_state.doc_meta = meta
                st.session_state.chat_history = []
                st.success(f"✦ {meta['chunks']} chunks indexed!")

    if st.button("🗑️ Clear"):
        for key in ["vector_store", "chat_history", "doc_meta", "raw_text", "chunks"]:
            st.session_state[key] = None if key != "chat_history" else []
        st.rerun()

# Main
if not st.session_state.vector_store:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem;">⚖️</div>
        <div style="font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; margin-bottom:0.5rem;">Upload a legal document to begin</div>
        <div style="color:#888; font-size:0.9rem;">Contracts, NDAs, Terms of Service, Legal briefs</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🔍 PII Detection", "Compares generic spaCy vs InLegalBERT — measures domain recall gap"),
        ("📋 Clause Classifier", "Identifies 15+ CUAD clause types using DeBERTa zero-shot"),
        ("🕸️ Argument Mining", "Maps Premise → Claim → Exception dependency graph"),
        ("🔒 Privacy Mode", "Toggle between Groq cloud and Ollama local inference")
    ]
    for col, (title, desc) in zip([col1,col2,col3,col4], features):
        with col:
            st.markdown(f'<div class="entity-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
else:
    meta = st.session_state.doc_meta
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">1</div><div class="stat-label">Contract</div></div>
        <div class="stat-box"><div class="stat-num">{meta['pages']}</div><div class="stat-label">Pages</div></div>
        <div class="stat-box"><div class="stat-num">{meta['chunks']}</div><div class="stat-label">Chunks</div></div>
        <div class="stat-box"><div class="stat-num">{len(st.session_state.chat_history)//2}</div><div class="stat-label">Queries</div></div>
    </div>
    """, unsafe_allow_html=True)

    # 4 TABS
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A", "🔍 PII Detection", "📋 Clause Classifier", "🕸️ Argument Mining"])

    # TAB 1: Q&A
    with tab1:
        st.markdown('<div class="section-label">✦ Legal Document Q&A</div>', unsafe_allow_html=True)

        if mode == "privacy":
            st.warning("🔒 Privacy Mode: Inference runs locally via Ollama. No data sent to cloud.")
        else:
            st.info("☁️ Cloud Mode: Using Groq API for fast inference.")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                latency = msg.get("latency", "")
                st.markdown(f'<div class="chat-ai">⚖️ {msg["content"]}<div class="chat-source">{msg.get("sources","")} {latency}</div></div>', unsafe_allow_html=True)

        quick = ["What are the payment terms?", "Who are the parties?", "What are termination conditions?", "List all obligations"]
        cols = st.columns(4)
        for i, q in enumerate(quick):
            with cols[i]:
                if st.button(q, key=f"q_{i}"):
                    st.session_state["pending_q"] = q
                    st.rerun()

        question = st.text_input("Ask about this contract:", placeholder="e.g. What is the governing law?")
        if st.session_state.get("pending_q"):
            question = st.session_state.pop("pending_q")

        if st.button("⚖️ Query", disabled=not question):
            with st.spinner(f"Querying via {mode} mode..."):
                docs = st.session_state.vector_store.similarity_search(question, k=4)
                context = "\n\n".join([d.page_content for d in docs])
                sources = list(set([
                    line.strip("[]") for d in docs
                    for line in d.page_content.split("\n")
                    if line.startswith("[PAGE")
                ]))
                sources_str = " | ".join([f"📍 {s}" for s in sources[:3]])

                result = query_with_benchmark(question, context, mode=mode)
                latency_str = f"⏱ {result['latency_seconds']}s ({result['model']})"

                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources_str,
                    "latency": latency_str
                })
                st.rerun()

    # TAB 2: PII Detection
    with tab2:
        st.markdown('<div class="section-label">✦ PII Entity Detection Benchmark</div>', unsafe_allow_html=True)
        st.markdown("Compares **generic spaCy** vs **InLegalBERT** on legal PII extraction.")

        sample_text = st.text_area(
            "Paste contract text to analyze:",
            value=st.session_state.raw_text[:1000] if st.session_state.raw_text else "",
            height=150
        )

        if st.button("🔍 Run PII Comparison"):
            with st.spinner("Running NER models..."):
                results = compare_ner_models(sample_text)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**spaCy (generic)**")
                st.metric("Entities found", results["spacy_count"])
                for etype, entities in results["spacy_entities"].items():
                    st.markdown(f"**{etype}:** {', '.join(entities[:5])}")

            with col_b:
                st.markdown("**InLegalBERT (domain-specific)**")
                st.metric("Entities found", results["legal_bert_count"])
                for ent in results["legal_bert_entities"][:10]:
                    st.markdown(f"**{ent.get('entity_group','?')}:** {ent.get('word','')}")

            st.success(f"📊 Research finding: {results['recall_gap']}")

    # TAB 3: Clause Classifier
    with tab3:
        st.markdown('<div class="section-label">✦ CUAD Clause Type Classifier</div>', unsafe_allow_html=True)
        st.markdown("Identifies clause types from 15 CUAD categories using DeBERTa zero-shot.")

        clause_text = st.text_area("Paste a clause to classify:", height=120,
            placeholder="e.g. Either party may terminate this agreement with 30 days written notice...")

        if st.button("📋 Classify Clause"):
            with st.spinner("Classifying..."):
                result = classify_clause(clause_text)

            st.markdown(f"**Top Clause Type:** `{result['top_clause_type']}`")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")

            if result.get("all_matches"):
                st.markdown("**All matches above threshold:**")
                for match in result["all_matches"]:
                    st.markdown(f"- `{match['label']}` — {match['score']*100:.1f}%")

        # Batch classify chunks
        if st.session_state.chunks and st.button("📋 Classify All Chunks"):
            with st.spinner(f"Classifying {min(10, len(st.session_state.chunks))} chunks..."):
                sample = st.session_state.chunks[:10]
                results = batch_classify(sample)

            for i, (chunk, result) in enumerate(zip(sample, results)):
                st.markdown(f"**Chunk {i+1}:** `{result.get('top_clause_type','?')}` ({result.get('confidence',0)*100:.1f}%)")
                st.caption(chunk[:150] + "...")

    # TAB 4: Argument Mining
    with tab4:
        st.markdown('<div class="section-label">✦ Argument Structure Mining</div>', unsafe_allow_html=True)
        st.markdown("Classifies clauses into **Premise → Claim → Exception → Obligation** structure.")

        if st.session_state.chunks and st.button("🕸️ Mine Arguments"):
            with st.spinner("Mining argument structure..."):
                sample_chunks = st.session_state.chunks[:8]
                mined = mine_arguments(sample_chunks)
                G = build_argument_graph(mined)
                summary = get_graph_summary(G)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Claims", summary["claims"])
            col2.metric("Premises", summary["premises"])
            col3.metric("Exceptions", summary["exceptions"])
            col4.metric("Graph Edges", summary["total_edges"])

            st.markdown("**Argument Components:**")
            colors = {"Claim": "🔴", "Premise": "🔵", "Exception": "🟡", "Obligation": "🟢", "Definition": "⚪"}
            for arg in mined:
                icon = colors.get(arg["component"], "⚫")
                st.markdown(f"{icon} **{arg['component']}** ({arg['confidence']*100:.0f}%) — {arg['text_preview']}")

        # Export
        if st.session_state.chat_history:
            chat_export = "\n\n".join([
                f"{'USER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}"
                for m in st.session_state.chat_history
            ])
            st.download_button("⬇ Export Chat", data=chat_export, file_name="legalrag_chat.txt", mime="text/plain")