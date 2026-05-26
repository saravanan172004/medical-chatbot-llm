# 🏥 MediBot — AI-Powered Medical Chatbot

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" width="120" alt="MediBot Logo"/>
</p>

<p align="center">
  <b>An end-to-end RAG-based Medical Chatbot powered by Google Gemini, Pinecone, LangChain & Flask</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/LangChain-0.2-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gemini-AI-orange?style=for-the-badge&logo=google"/>
  <img src="https://img.shields.io/badge/Pinecone-VectorDB-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Render-Deployed-blue?style=for-the-badge&logo=render"/>
</p>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Data Ingestion](#-data-ingestion)
- [Running the App](#-running-the-app)
- [Deployment on Render](#-deployment-on-render)
- [How It Works](#-how-it-works)
- [Screenshots](#-screenshots)
- [Disclaimer](#-disclaimer)

---

## 📖 About the Project

**MediBot** is an AI-powered medical assistant chatbot built using **Retrieval-Augmented Generation (RAG)**. It reads medical PDF documents, stores them as vector embeddings in **Pinecone**, and uses **Google Gemini** to answer user queries with context-aware, accurate responses.

> ⚠️ MediBot provides general medical information only. Always consult a licensed physician for medical advice.

---

## ✨ Features

- 🤖 **AI-Powered Q&A** — Ask any medical question and get intelligent answers
- 📚 **RAG Architecture** — Retrieves relevant context from medical PDFs before answering
- 🔍 **Semantic Search** — Uses vector embeddings for accurate document retrieval
- 💊 **Quick Topics** — Pre-built shortcuts for Symptoms, Medications, Nutrition, Heart Health, Sleep, Mental Health
- 🎨 **Beautiful UI** — Modern dark-themed chat interface with sidebar navigation
- 🌐 **Cloud Deployed** — Hosted on Render.com with free tier
- 🔒 **Secure** — API keys stored as environment variables, never in code

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML, CSS, JavaScript, jQuery |
| **Backend** | Python, Flask |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database** | Pinecone (Serverless, AWS us-east-1) |
| **RAG Framework** | LangChain |
| **PDF Loader** | PyPDF + LangChain DirectoryLoader |
| **Deployment** | Render.com |
| **Environment** | Conda (Python 3.10) |

---

## 📁 Project Structure

```
medical-chatbot-llm/
├── Data/                        # Medical PDF files (not pushed to GitHub)
│   └── medical_book.pdf
├── research/
│   └── trails.ipynb             # Jupyter notebook for experimentation
├── src/
│   ├── __init__.py
│   ├── helper.py                # PDF loading, text splitting, embeddings
│   └── prompt.py                # System prompt for Gemini
├── static/
│   └── style.css                # Chat UI styles
├── templates/
│   └── chat.html                # Chat UI template
├── .env                         # API keys (never push to GitHub)
├── .gitignore
├── app.py                       # Flask application
├── store_index.py               # Data ingestion to Pinecone
├── requirements.txt
├── render.yaml                  # Render deployment config
├── runtime.txt                  # Python version for Render
└── README.md
```

---

## ✅ Prerequisites

Before you begin, make sure you have:

- Python 3.10 installed
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- A [Pinecone](https://app.pinecone.io) account (free)
- A [Google AI Studio](https://aistudio.google.com) account (free)
- A [GitHub](https://github.com) account
- Git installed

---

## 🚀 Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/saravanan172004/medical-chatbot-llm.git
cd medical-chatbot-llm
```

### Step 2 — Create and activate conda environment

> ⚠️ Use **Anaconda Prompt** on Windows, not Git Bash

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### Where to get the API keys:

| Key | Website |
|---|---|
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) → Get API Key |

> 🔒 Never push your `.env` file to GitHub. It is already in `.gitignore`.

---

## 📥 Data Ingestion

### Step 1 — Add your medical PDF

Create a `Data/` folder and add your medical PDF files:

```
medical-chatbot-llm/
└── Data/
    └── your_medical_book.pdf
```

Free medical PDFs:
- [nlm.nih.gov](https://www.nlm.nih.gov) — National Library of Medicine
- [pdfdrive.com](https://www.pdfdrive.com) — Search "medical textbook"

### Step 2 — Create Pinecone index and upload embeddings

```bash
conda activate medibot
python store_index.py
```

This will:
1. Load all PDFs from the `Data/` folder
2. Split them into text chunks
3. Generate HuggingFace embeddings
4. Upload everything to Pinecone

> ⏳ This may take a few minutes depending on PDF size.

---

## ▶️ Running the App

```bash
conda activate medibot
python app.py
```

Open your browser and go to:

```
http://localhost:8000
```

---

## ☁️ Deployment on Render

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "ready for deployment"
git push origin main
```

### Step 2 — Create account on Render

Go to [render.com](https://render.com) → Sign up with GitHub

### Step 3 — Create a new Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your `medical-chatbot-llm` repository
3. Fill in the settings:

| Field | Value |
|---|---|
| **Name** | medical-chatbot |
| **Region** | Singapore |
| **Branch** | main |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --host=0.0.0.0 --port=8000` |
| **Plan** | Free |

### Step 4 — Add Environment Variables

Go to **Environment** tab and add:

| Key | Value |
|---|---|
| `PINECONE_API_KEY` | your pinecone key |
| `GOOGLE_API_KEY` | your gemini key |
| `PYTHON_VERSION` | `3.11.4` |

### Step 5 — Deploy!

Click **"Create Web Service"** — your app will be live at:

```
https://medical-chatbot-xxxx.onrender.com
```

> ⚠️ Free tier apps spin down after 15 min of inactivity. First load may take ~30 seconds.

---

## ⚙️ How It Works

```
User Question
      ↓
Flask App (app.py)
      ↓
LangChain RAG Chain
      ↓
HuggingFace Embeddings → Pinecone Vector Search
      ↓
Top 3 Relevant Document Chunks Retrieved
      ↓
Google Gemini LLM (gemini-2.5-flash)
      ↓
Context-Aware Medical Answer
      ↓
User
```

### RAG Pipeline:

1. **Ingestion** — PDFs are loaded, split into 500-character chunks with 20-character overlap
2. **Embedding** — Each chunk is converted to a 384-dimension vector using HuggingFace
3. **Storage** — Vectors stored in Pinecone serverless index
4. **Retrieval** — User query is embedded and top-3 similar chunks are retrieved
5. **Generation** — Gemini generates an answer using retrieved context + system prompt

---

## 🖼️ Screenshots

### Chat Interface
![MediBot UI](https://cdn-icons-png.flaticon.com/512/387/387569.png)

- Dark themed medical UI
- Left sidebar with quick topic navigation
- Real-time typing indicator
- Suggestion chips for common medical queries
- Disclaimer for medical advice

---

## ⚠️ Disclaimer

> MediBot is an AI-powered tool designed for **educational and informational purposes only**.
> It is **not a substitute** for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare provider for medical concerns.

---

## 👨‍💻 Author

**Saravanan S**

- GitHub: [@saravanan172004](https://github.com/saravanan172004)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ using LangChain, Gemini & Pinecone</p>
