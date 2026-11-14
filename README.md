# Palm Mind RAG Backend

I built a FastAPI backend with two REST APIs.

The first one is a **document ingestion API**. You upload a PDF, TXT or DOCX file, pick a chunking strategy (recursive or semantic), and the app extracts the text, cleans it, splits it into chunks, turns the chunks into embeddings, and stores them in Qdrant. The document's details are saved in PostgreSQL.

The second one is a **conversational RAG API**. You chat with it and it answers questions from your uploaded documents, using search I wrote by hand (dense search plus BM25, merged with RRF and reranked). It remembers the conversation in Redis, so follow-up questions work. The same chat endpoint can also book an interview: it collects your name, email, date and time from normal messages, checks them, and saves the booking in PostgreSQL.

A small router decides what each message is about, so the app only searches your documents when it needs to.

---

## Folder structure

```
palm-mind-rag/
├── app/
│   ├── main.py             # starts the app, creates tables, loads models and connections once
│   ├── config.py           # settings from .env
│   ├── dependencies.py
│   ├── schemas.py          # request and response types
│   ├── db.py               # PostgreSQL tables and save functions
│   ├── routes/
│   │   ├── ingestion.py    # POST /documents/upload
│   │   └── chat.py         # POST /chat
│   └── services/
│       ├── embedder.py
│       ├── vector_store.py # Qdrant
│       ├── memory.py       # Redis
│       ├── llm.py          # Groq
│       ├── extraction.py   # PDF, TXT, DOCX (OCR optional)
│       ├── cleaning.py
│       ├── chunking.py     # recursive and semantic
│       ├── ingestion.py    # the full upload flow
│       ├── retrieval.py    # dense search, BM25, RRF
│       ├── reranking.py    # reranker and relevance cut-off
│       ├── generation.py   # question rewriting, prompt, answer
│       ├── rag.py          # RAG steps in order
│       ├── router.py       # detects intent, picks the workflow
│       └── booking.py      # kept separate from RAG
├── scripts/
│   └── cli.py              # optional terminal tester
├── tests/
├── docker-compose.yml      # Qdrant, Redis, PostgreSQL
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Original uploaded files are saved in `storage/uploads/`, which is created automatically when the app starts and is not committed to Git.

---

## System design

**Document ingestion**

1. Upload a file and choose `recursive` or `semantic` chunking.
2. Check the file (type, size, not empty) and save the original to disk.
3. Extract the text. PDFs are read page by page so page numbers are kept.
4. Clean the text (broken characters, extra spaces).
5. Split it into chunks with the chosen strategy.
6. Turn each chunk into an embedding.
7. Store the vectors, chunk text and details in Qdrant.
8. Save the document's details in PostgreSQL.
9. Rebuild the BM25 keyword index and return the document ID.

Recursive chunking splits at paragraphs, then lines, then sentences. Semantic chunking compares each sentence with the next and starts a new chunk when the topic changes.

**Chat (including booking)**

1. Receive the message and session ID (a new session is made if none is sent).
2. Load the last few messages and any unfinished booking from Redis.
3. The router picks one intent: document question, follow-up, interview booking, booking follow-up, or general chat.
4. For a **document question**: rewrite follow-ups into full questions, search with dense search and BM25, merge with RRF, rerank, keep chunks above the cut-off, and let Groq answer from them. If nothing is relevant enough, it says so.
5. For **booking**: Groq pulls out name, email, date and time, the app validates them, asks for anything missing, then saves the finished booking in PostgreSQL. Unfinished bookings wait in a Redis Hash.
6. For **general chat**: Groq replies, with no document search.
7. Save the messages and sources in Redis, and return the reply.

---

## Tech stack

- **FastAPI**: it is async, typed, and builds a test page for free, while Flask needs add-ons for all of that and Django is far too heavy for two endpoints.
- **PyMuPDF and python-docx**: PyMuPDF reads PDFs fast and page by page, which older libraries like PyPDF2 often do worse, and python-docx covers Word files.
- **ftfy**: PDFs produce broken characters, and a plain regex cannot repair those.
- **Sentence Transformers (BAAI/bge-base-en-v1.5)**: strong quality that runs on a normal laptop, unlike paid embedding APIs that cost money and need the internet on every call.
- **Qdrant**: it runs with one Docker command and stores text and details next to each vector, while FAISS is only an index library with no storage or filtering.
- **BM25 (rank_bm25)**: it finds exact names and terms, which vector search alone often misses.
- **RRF**: it merges two ranked lists without tuning, unlike mixing raw scores, which breaks because the two searches score on different scales.
- **Reranker (BAAI/bge-reranker-base)**: it reads the question and chunk together, so it is far more accurate than trusting vector similarity alone.
- **Groq**: very fast replies, which matters because one chat message can use the LLM several times (route, rewrite, answer), and a slower host would feel sluggish.
- **PostgreSQL and SQLAlchemy**: bookings and document records have a fixed shape, so SQL fits better than a document database, and SQLite struggles when several users write at once. The tables are created automatically when the app starts.
- **Redis**: it has built-in expiry plus Lists and Hashes that match chat history and unfinished bookings, while an in-memory dictionary loses everything on restart.
- **Docker Compose**: one command starts all three databases, instead of installing each by hand.

---

## How to run it

**You need:** Python 3.11 or newer, Docker Desktop (running), Git, and a Groq API key with a model name from your Groq account.

1. Get the code:
```
git clone <your-repo-url>
cd palm-mind-rag
```

2. Create and activate a virtual environment.

Windows:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Mac or Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the packages (this takes a few minutes):
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

4. Make your settings file, then open `.env` and fill in `GROQ_API_KEY` and `GROQ_MODEL` (no quotes, no spaces around the `=`):

Windows: `Copy-Item .env.example .env`
Mac or Linux: `cp .env.example .env`

5. Start the databases:
```
docker compose up -d
```

6. Start the app. The first start is slow because the models download, and the database tables are created automatically:
```
uvicorn app.main:app
```

7. Open **http://localhost:8000/docs**. You can test everything from this page by clicking buttons:
   - `POST /documents/upload`: choose a file and a `chunking_strategy`.
   - `POST /chat`: send `{"message": "your question"}`. The reply includes a `session_id`. Paste it into your next message to keep the same conversation, or leave it out to start a new one.
   - To book an interview, send "I want to book an interview" and then give your details, using the same `session_id`.

To run the tests: `python -m pytest`

To try the app from the terminal instead of the browser: `python scripts/cli.py`

---

## Settings

Everything is in `.env`, and the app needs a restart after any change. The ones you might care about:

- `MAX_UPLOAD_MB`: largest file allowed (10).
- `ENABLE_OCR`: reads scanned PDF pages, off by default.
- Chunk sizes: `RECURSIVE_CHUNK_SIZE` (1000) and `SEMANTIC_MAX_CHARS` (1000).
- `RELEVANCE_THRESHOLD`: how high a chunk must score to be used (0.3). Lower it if good questions get "not found".
- `FINAL_TOP_K`: how many chunks go into the answer (5).
- `CHAT_TTL_SECONDS` and `BOOKING_TTL_SECONDS`: how long chats (24 hours) and unfinished bookings (1 hour) are kept.

---

## Evidence

**Document ingestion**

> **[Screenshot placeholder]** Caption: Uploading a document with `recursive` chunking and the 201 response showing the chunk count.

> **[Screenshot placeholder]** Caption: Uploading a document with `semantic` chunking and the 201 response showing the chunk count.

**Chat**

> **[Screenshot placeholder]** Caption: A sample question answered from the document, with its source references.

> **[Screenshot placeholder]** Caption: The uvicorn terminal log showing the reranker scores for that question, with the best chunk above the 0.3 cut-off.

**Interview booking**

> **[Screenshot placeholder]** Caption: A booking completed over several messages, ending with `booking_status: completed`.

> **[Screenshot placeholder]** Caption: The saved booking in the PostgreSQL `bookings` table.

**Automated tests**

> **[Screenshot placeholder]** Caption: Output of `python -m pytest` with all tests passing.