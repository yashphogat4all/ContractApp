import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Local embeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama # Local LLM
from langchain_classic.chains import RetrievalQA

# Page Setup
st.set_page_config(page_title="Insurance Training Assistant", page_icon="🛡️")
st.title("🛡️ Insurance Assistant (Local AI & HuggingFace)")
# --- 1. Load and Process Data ---
@st.cache_resource
def load_and_process_data():
    try:
        st.info("Downloading REAL Q&A dataset from Hugging Face...")
        
        # A. Your exact dictionary setup
        splits = {'train': 'train.jsonl', 'validation': 'valid.jsonl', 'test': 'test.jsonl'}
        dataset_url = "hf://datasets/deccan-ai/insuranceQA-v2/" + splits["train"]
        
        # B. The corrected read_json line (with the 500 row limit for speed)
        df = pd.read_json(dataset_url, lines=True).head(500)
        
        # C. The Bulletproof Stitch
        df["combined_text"] = df.fillna("").astype(str).agg('\n'.join, axis=1)
        
        loader = DataFrameLoader(df, page_content_column="combined_text")
        documents = loader.load()
                
        # D. Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
                
        # E. Create Embeddings & Vector Store (Locally!)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
                
        return vectorstore
            
    except Exception as e:
        st.error(f"Failed to process data: {e}")
        return None

# Initialize the Knowledge Base
vectorstore = load_and_process_data()

# --- 2. Chat Interface ---
if vectorstore:
    # Set up Ollama (Make sure you ran 'ollama run llama3.2' in terminal first!)
    llm = ChatOllama(model="llama3.2", temperature=0)
        
    # Create the Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the insurance policy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking locally..."):
                try:
                    response = qa_chain.invoke(prompt)
                    st.markdown(response['result'])
                    st.session_state.messages.append({"role": "assistant", "content": response['result']})
                except Exception as e:
                    st.error(f"Chat failed: {e}")
                    st.info("💡 Make sure Ollama is installed and running in the background!")

else:
    st.warning("⚠️ System halted. Fix the dataset loading issue above to proceed.")
