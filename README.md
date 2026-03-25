# 🎬 YouTube AI Assistant

A RAG-powered assistant that answers questions about any YouTube video using its transcript. Built with LangChain, Groq, and Streamlit.

## How It Works

1. Paste a YouTube URL
2. Ask a question about the video
3. The app fetches the transcript, searches for relevant sections, and uses an LLM to generate an answer

Under the hood it uses **Retrieval Augmented Generation (RAG)** — the transcript is split into chunks, converted into vector embeddings, stored in a FAISS vector database, and the most relevant chunks are retrieved and passed to the LLM to answer your question.

## Tech Stack

- **LangChain** — RAG pipeline and chaining
- **Groq (LLaMA 3.3)** — LLM for answer generation
- **HuggingFace Inference API** — text embeddings
- **FAISS** — vector store for similarity search
- **Streamlit** — web interface

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/youtube-ai-assistant.git
cd youtube-ai-assistant
```

**2. Create and activate a virtual environment**
```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash)
```

**3. Install dependencies**
```bash
pip install langchain-groq langchain-community langchain-text-splitters langchain-huggingface faiss-cpu youtube-transcript-api python-dotenv streamlit
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
```

- Get your Groq API key at [console.groq.com](https://console.groq.com)
- Get your HuggingFace API key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (set type to Read, enable Inference Providers)

**5. Run the app**
```bash
streamlit run main.py
```

## Project Structure

```
youtube-ai-assistant/
├── main.py              # Streamlit UI
├── langchain_helper.py  # RAG pipeline logic
├── .env                 # API keys (not committed)
├── .gitignore
└── README.md
```

## Author

Built by Mowa — medical student and AI engineering enthusiast.
