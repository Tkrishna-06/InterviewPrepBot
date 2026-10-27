# 🤖 AI Technical Interview Preparation ChatBot (Advanced RAG Pipeline)

## 📌 Project Overview
This project is an Enterprise-Grade Conversational AI Chatbot designed to help candidates prepare for technical interviews. Built using a **Retrieval-Augmented Generation (RAG)** architecture, the system scraps data from GeeksForGeeks across DSA, System Design, CS Fundamentals, and HR Q&A, indexes it locally, and delivers highly accurate, context-grounded responses. 

This repository fully implements the core requirements alongside **both bonus criteria** (Hindi/Hinglish Language Support.

---

## ⚡ Key Features & Architecture
*   **Dual-Model Intelligence**: Employs `llama-3.3-70b-versatile` via Groq for high-reasoning QA execution, alongside a lightweight `llama-3.1-8b-instant` model dedicated exclusively to asynchronous chat summarization to lower token overhead.
*   **State-of-the-Art Local Embeddings**: Utilizes the high-performance open-source `BAAI/bge-base-en-v1.5` model to capture subtle semantics in code blocks and technical terminology.
*   **Maximal Marginal Relevance (MMR)**: Search queries look for diverse chunks (`k=5`, `fetch_k=20`) to eliminate redundancy and prevent the bot from missing content due to arbitrary text splitting.
*   **Decoupled Dual-State Memory**: Uses an independent UI thread state alongside a compressed background sliding-window `llm_history` to ensure the application *never* hits token payload limits or crashes during prolonged conversations.
*   **Recency Bias Prompting**: Structural rules regarding anti-hallucination and language enforcement are injected directly at the human question boundary to maintain strict alignment in smaller context windows.

---

## 🛠️ Detailed Pipeline Flow
1. **Data Ingestion**: `scraper_indexer.py` handles parsing targeted GeeksForGeeks URLs using LangChain documentation loaders.
2. **Chunking Strategy**: Documents are partitioned via `RecursiveCharacterTextSplitter` with an optimized configuration of `chunk_size=2000` and `chunk_overlap=400` to maintain the integrity of multi-line C++/Java/Python snippets.
3. **Local Storage**: Vectors are committed into a persistent local `ChromaDB` instance (`./chroma_db`).
4. **History-Aware Retrieval**: The incoming prompt runs through an LCEL pipeline where a `contextualize_q_prompt` reformulates follow-up queries into self-contained standalone statements based on historic conversational context.
5. **Guardrailed Generation**: Text generation strictly operates under a Zero-Hallucination mandate inside defined `<context>` fences. Missing answers fail gracefully to pre-configured context exceptions.

---

## 📁 Project Folder Structure
```text
├── chroma_db/               # Persistent local vector database storage (Git-ignored)
├── scraper_indexer.py       # Web scraping, text processing, and indexing script
├── rag_engine.py            # Main RAG engine and LangChain LCEL pipeline compilation
├── app.py                   # Streamlit Frontend Web Dashboard UI 
├── Requirements.txt         # Complete Python dependency configuration
├── Dockerfile               # Multi-stage image assembly configuration
├── .gitignore               # Strict file ignore mappings (.env, vector files, cache)
└── README.md                # Project documentation and architectural manifest

🚀 How to Run Locally
1. Configure the Environment
Create a .env file in your root folder and add your Groq API access token:

Code snippet
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
2. Install Project Dependencies
Initialize a virtual environment and load the verified project requirements:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
3. Ingest and Build Vector Database
Execute the scraping pipeline to build the local vector database index:

Bash
python scraper_indexer.py
Note: Ensure the output displays "Split documents into X chunks" and successfully saves to ./chroma_db.

4. Fire Up the Streamlit UI
To launch the primary web-based application panel:

Bash
streamlit run app.py
