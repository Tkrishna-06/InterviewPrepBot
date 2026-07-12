import os
import json

os.environ["USER_AGENT"] = "SameerInterviewBot/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings

# --- NEW CONFIGURATION LOGIC ---
# Open the config file and load the URLs dynamically
with open("config.json", "r") as config_file:
    config_data = json.load(config_file)
    URLS_TO_SCRAPE = config_data["urls_to_scrape"]

DB_DIR = "./chroma_db"

def build_vector_database():
    print("Starting Web Scraping...")
    loader = WebBaseLoader(URLS_TO_SCRAPE)
    documents = loader.load()
    print(f"Scraped {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print("Downloading local embeddings model and indexing database...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"Success! Vector DB saved locally at {DB_DIR}")

if __name__ == "__main__":
    build_vector_database()