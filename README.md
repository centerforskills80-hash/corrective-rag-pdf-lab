# Corrective RAG (CRAG) PDF Lab

A beginner-friendly Retrieval-Augmented Generation project for an AI class. Upload multiple PDFs, index their text, and ask questions with source/page references.

## What this project does

1. Reads text from multiple PDFs.
2. Splits pages into overlapping chunks.
3. Builds a local TF-IDF search index (no paid embedding service).
4. Retrieves likely relevant chunks for a question.
5. Applies a simple corrective step when the best match is weak: it broadens retrieval and checks for non-zero matches.
6. Sends the selected context to a language model (Ollama locally or Groq API).

**Important:** This is an educational CRAG-style demo, not a full reproduction of the research paper. TF-IDF scores are heuristic relevance signals. It does not fine-tune/train the language model; it indexes your PDFs for retrieval.

## Free model options

### Option A — Ollama locally (recommended for the lab)
- No API token required.
- Model inference runs on your computer.
- No hosted token quota, but your computer's hardware and disk space are limits.
- Ollama model downloads use disk space and internet once.

Install Ollama from https://ollama.com, then open a terminal and run:

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

Leave Ollama running while using the app. If your computer is low on RAM, choose a smaller model available in Ollama.

### Option B — Groq API (for cloud-hosted Streamlit)
- Create an account/key through https://console.groq.com/
- Enter the API key in the app sidebar.
- Free access, model availability, and rate limits may change. It is **not unlimited**.
- Never commit your API key to GitHub.

## Run locally on Windows

Python 3.10 or 3.11 is a good choice for a beginner setup.

1. Install Python and Git.
2. Download/clone this project.
3. Open the project folder in VS Code.
4. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

5. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Install/start Ollama and pull a model (Option A above).
7. Launch the app:

```powershell
streamlit run app.py
```

Your browser should open the Streamlit interface, usually at `http://localhost:8501`.

## Upload to GitHub

1. Create a new **public** GitHub repository, for example `corrective-rag-pdf-lab`.
2. Extract this project ZIP.
3. In GitHub, choose **Add file → Upload files** and upload `app.py`, `rag_engine.py`, `requirements.txt`, `README.md`, and `.gitignore`.
4. Commit the files.
5. Do not upload private PDFs, `.env` files, or API keys.

Or use Git in the project folder:

```bash
git init
git add app.py rag_engine.py requirements.txt README.md .gitignore
git commit -m "Add corrective RAG PDF lab"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/corrective-rag-pdf-lab.git
git push -u origin main
```

Replace the example URL with your own repository URL.

## Deploy on Streamlit Community Cloud

Streamlit Community Cloud can host the Streamlit app, but it does not automatically run your local Ollama service. For a cloud deployment, use the Groq API option or configure another model endpoint you control.

1. Push the files to GitHub.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Deploy the app.
6. Open the app URL. In the sidebar, select **Groq API** and enter your API key. If the deployment's secrets workflow is available, store keys in Streamlit Secrets instead of typing them into a shared session. Never put secrets in source code.
7. Test with a non-sensitive sample PDF.

### Important cloud deployment notes
- Free hosting has CPU, memory, storage, and usage limits.
- Uploaded files and the index in this demo live in session memory and may disappear when the app restarts or the session ends.
- Do not upload confidential or personal documents to a public demo.
- The default Groq model name in the app may change availability. If it is unavailable, choose a currently supported model in the provider's console and enter its exact model ID.
- There is no guaranteed unlimited hosted free GPT/API option. Local Ollama avoids API token quotas but is limited by your own hardware.

## How to test your lab

Try questions that:
- Have a clear answer in one PDF.
- Need information from two different PDFs.
- Are not answered by any uploaded PDF (the app should say it cannot find enough evidence).
- Use different wording from the document.

## Project files

- `app.py` — Streamlit front end.
- `rag_engine.py` — PDF processing, chunking, retrieval, corrective step, and model call.
- `requirements.txt` — Python dependencies.
- `README.md` — setup and deployment instructions.
- `.gitignore` — common files to keep out of Git.
