# 🛡️ Insurance Assistant (Local AI & HuggingFace)

Welcome to **ContractApp**! This is a fully local, Retrieval-Augmented Generation (RAG) chatbot designed to act as an Insurance Training Assistant. 

Instead of relying on expensive cloud APIs, this project is built to run **100% free and locally** on your machine. It pulls a training dataset directly from Hugging Face, processes it into a local vector database, and uses Ollama to answer questions—all through a clean Streamlit web interface.

## ✨ Features
* **100% Local & Free:** No OpenAI or paid API keys required.
* **Direct Dataset Loading:** Bypasses complex web scraping by directly reading `.parquet` files from Hugging Face into a Pandas DataFrame.
* **Local Embeddings:** Uses Hugging Face's `all-MiniLM-L6-v2` via CPU to create text embeddings.
* **Local LLM:** Powered by `Llama 3.2` (via Ollama) for private, offline inference.
* **Interactive UI:** A conversational chat interface built with Streamlit.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Processing:** Pandas, PyArrow
* **Orchestration:** LangChain, Langchain-Classic
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **Vector Store:** FAISS (CPU)
* **Local AI Engine:** Ollama

## 🚀 Getting Started

### Prerequisites
1. **Python 3.8+** installed on your machine.
2. **Ollama** installed ([Download here](https://ollama.com/)).

### Installation

1. **Clone the repository (or create your project folder):**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/contractapp.git](https://github.com/YOUR-USERNAME/contractapp.git)
   cd contractapp