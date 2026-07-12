import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ONLY using langchain_core (bypassing the broken langchain package completely)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Load API keys from .env
load_dotenv()

# Configure the Streamlit page
st.set_page_config(page_title="Interview Prep Bot", page_icon="🤖")

@st.cache_resource
def init_rag_pipeline():
    """Initializes the RAG pipeline using pure LCEL with strict token limits."""
    if not os.path.exists("./chroma_db"):
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Context cap to prevent oversized token payloads
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,          
            "fetch_k": 20   
        }
    )
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    # 1. Contextualize the question based on chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    
    # History-Aware Retriever pipeline
    history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser() | retriever

    # 2. Final QA Prompt (Optimized for 8B Attention Span)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict technical interview preparation assistant. 
        You operate under a strict ZERO-HALLUCINATION policy.

        CRITICAL RULES:
        1. You MUST answer the user's question EXCLUSIVELY using the information provided in YOUR RETRIEVED KNOWLEDGE BASE below. 
        2. If the user asks about a topic or concept that is NOT explicitly mentioned in the text below, you MUST output EXACTLY this phrase and nothing else: 
        "I'm sorry, but I don't have enough information in my retrieved context to answer that accurately."
        3. Treat every new question as a brand new topic. Do not assume the current question is related to the previous chat history.
        4. Do NOT use your general pre-trained knowledge. Do NOT write code unless it is in the context.

        YOUR RETRIEVED KNOWLEDGE BASE:
        <context>
        {context}
        </context>"""),
        
        MessagesPlaceholder("chat_history"),
        
        # Notice how we put the language rule down here! This is Recency Bias Injection.
        ("human", """User Question: ```{question}``` 

    CRITICAL REMINDERS BEFORE YOU ANSWER:
    1. STRICT LANGUAGE ISOLATION: Look ONLY at the User Question above. If it is in English, your ENTIRE reply MUST be in English. If it is in Hindi, your ENTIRE reply MUST be in Hindi. Do NOT copy the language of the chat history.
    2. Answer directly. Do not start with "According to the context" or "Based on my knowledge base."
    3. Keep technical terms (like 'Stack', 'Queue', 'Time Complexity') in English script."""),
    ])

    # Truncate character string length to prevent oversized context payloads
    def format_docs(docs):
        combined_text = "\n\n".join(doc.page_content for doc in docs)
        return combined_text[:15000]

    # 3. Build the final chain using pure LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(history_aware_retriever.invoke(x)))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- UI Setup ---
st.title("🤖 Tech Interview Prep Bot")
st.write("Ask me anything about DSA, System Design, CS Fundamentals, or HR questions!")

# Initialize the pipeline
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = init_rag_pipeline()
    # Create a separate, lightweight LLM instance just for summarization
    st.session_state.summary_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)

# Stop execution if DB is missing
if st.session_state.rag_chain is None:
    st.error("Vector DB not found. Please run `python scraper_indexer.py` first to scrape the data.")
    st.stop()

# ==========================================
# 1. MEMORY INITIALIZATION
# ==========================================
# The UI Memory (Keeps everything so the screen never glitches)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ready to help you prepare for your technical interview. What topic would you like to cover?"}
    ]
    
# The Bot's Brain (Keeps only recent stuff to prevent API crashes)
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []
    
if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = ""

# ==========================================
# 2. UI RENDERING
# ==========================================
# Draw all messages. Because we never delete from this list, ghosting is impossible!
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 3. CHAT LOGIC
# ==========================================
if prompt := st.chat_input("E.g., What is the difference between a stack and a queue?"):
    
    # Save the user's prompt to BOTH memories
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.llm_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Put EVERYTHING inside the standard, normal spinner next to the bot's face
        with st.spinner("Thinking..."):
            try:
                # --- MEMORY COMPRESSION (Happens silently in the background!) ---
                if len(st.session_state.llm_history) > 4:
                    # Grab older messages
                    msgs_to_summarize = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in st.session_state.llm_history[:-2]]
                    )
                    
                    summary_prompt = f"""
                    Combine the Current Summary with these New Messages into a short paragraph.
                    Current Summary: {st.session_state.chat_summary}
                    New Messages: {msgs_to_summarize}
                    """
                    # Update summary
                    st.session_state.chat_summary = st.session_state.summary_llm.invoke(summary_prompt).content
                    
                    # Shrink ONLY the Bot's memory. We leave the UI memory alone!
                    st.session_state.llm_history = st.session_state.llm_history[-2:]

                # --- BUILD THE RAG CONTEXT ---
                chat_history = []
                
                if st.session_state.chat_summary:
                    chat_history.append(AIMessage(content=f"[SUMMARY SO FAR: {st.session_state.chat_summary}]"))
                
                # Only feed the last 2-3 compressed messages to the API
                for msg in st.session_state.llm_history: 
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"][:300]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"][:500]))

                # --- GENERATE RESPONSE ---
                answer = st.session_state.rag_chain.invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })
                
            except Exception as e:
                answer = f"An error occurred: {str(e)}"
                
        # Display the final answer and save it to BOTH memories
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.llm_history.append({"role": "assistant", "content": answer})