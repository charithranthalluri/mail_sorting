# AI-Powered Gmail Sorting and Automation System

A production-quality, modular Python application that continuously monitors a Gmail mailbox, extracts and cleans incoming messages, interprets and classifies them with an LLM via strict JSON schema validation, applies deterministic Python decision logic, and performs safe, idempotent Gmail actions (labeling, starring).

---

## Architecture & Data Flow

The system is designed with strict separation of concerns following the rule:

> **AI interprets → Python validates → Python decides → Python executes.**

The LLM is strictly an interpretation engine; it is never granted direct execution privileges.

```
       Gmail API
           ↓
     Email Watcher / Reader (gmail/reader.py)
           ↓
     Email Parser (processing/parser.py)
           ↓
     Email Cleaner (processing/cleaner.py)
           ↓
     Normalized EmailMessage (storage/models.py)
           ↓
     LLM Classifier (ai/classifier.py + ai/prompts.py)
           ↓
     Strict Schema Validation (ai/schema.py)
           ↓
     Deterministic Decision Engine (engine/decision_engine.py)
           ↓
     Gmail Action Executor (gmail/actions.py)
           ↓
     SQLite Persistent State (storage/database.py)
           ↓
     Audit Trail & Structured Logging (utils/logger.py)
```

---

## Project Structure

```
mail_sorter/
│
├── main.py                  # Orchestrator & continuous polling loop
├── config.py                # Centralized configuration with environment parsing
├── requirements.txt         # Production and development dependencies
├── README.md                # Documentation and setup guide
├── .env.example             # Template for environment configuration
├── .gitignore               # Excludes secrets, credentials, tokens, DBs, and virtualenvs
│
├── gmail/                   # Gmail API integration
│   ├── __init__.py
│   ├── auth.py              # OAuth2 authentication lifecycle & token auto-refresh
│   ├── reader.py            # Message querying, retrieval, pagination & exponential backoff
│   └── actions.py           # Safe, idempotent label mutations, starring, archiving & caching
│
├── processing/              # Email parsing and cleaning
│   ├── __init__.py
│   ├── parser.py            # MIME decoding (RFC 2047, multipart, base64url)
│   ├── cleaner.py           # Boilerplate, signature, and quote removal; HTML to text
│   └── attachments.py       # Non-destructive attachment metadata extraction
│
├── ai/                      # AI classification and schema validation
│   ├── __init__.py
│   ├── schema.py            # Pydantic models for categories, priorities, urgencies & validation
│   ├── prompts.py           # Versioned system prompts and formatted email context builders
│   └── classifier.py        # Abstract classifier interface + Gemini implementation with retry
│
├── engine/                  # Deterministic decision making
│   ├── __init__.py
│   └── decision_engine.py   # Rule evaluation mapping classifications to concrete actions
│
├── storage/                 # Persistence and state machine
│   ├── __init__.py
│   ├── models.py            # Normalized EmailMessage & ProcessingRecord models
│   └── database.py          # SQLite with WAL mode, parameterized queries & state tracking
│
├── utils/                   # Shared utilities
│   ├── __init__.py
│   └── logger.py            # Contextual logging with per-email trace IDs & safe redaction
│
└── tests/                   # 135+ unit and integration tests (zero external network dependency)
    ├── __init__.py
    ├── test_parser.py       # MIME decoding, plain text, HTML, and attachments
    ├── test_cleaner.py      # Signatures, quoted replies, whitespace, and HTML tags
    ├── test_classifier.py   # Schema validation, edge cases, and mocked LLM calls
    ├── test_decision_engine.py # Rule matching, priorities, urgencies, and stars
    ├── test_database.py     # SQLite operations, state transitions, idempotency & retries
    ├── test_actions.py      # Dry-run vs live actions, label creation, and error resilience
    ├── test_reader.py       # Query building, pagination, and retry on 429/500/503
    ├── test_attachments.py  # Attachment parsing and human-readable summarization
    ├── test_prompts.py      # Versioning, schema prompts, and body truncation
    ├── test_auth.py         # OAuth2 token loading, saving, and refresh handling
    └── test_main.py         # End-to-end integration test of the application pipeline
```

---

## Key Features

1. **Production-Ready MIME Parsing**:
   - Handles `text/plain`, `text/html`, `multipart/alternative`, `multipart/mixed`, and arbitrarily nested MIME trees.
   - Decodes base64url bodies without padding errors.
   - Decodes RFC 2047 encoded headers (e.g. `Subject: =?UTF-8?B?...?=`).
   - Extracts attachment metadata (filename, size, MIME type, ID) without downloading heavy payloads.

2. **Email Content Cleaning**:
   - Converts HTML to readable text using BeautifulSoup, preserving links and formatting.
   - Strips email signatures (`--`, `Sent from my iPhone`, `Get Outlook for iOS`, etc.).
   - Truncates quoted reply chains (`On ... wrote:`, `> quote`).
   - Cleans tracking pixels and excessive whitespace.
   - Preserves critical details: deadlines, dates, instructions, sender names.

3. **Strict AI Schema Validation**:
   - Allowed categories: `college`, `work`, `personal`, `finance`, `shopping`, `social`, `newsletter`, `promotion`, `notification`, `spam`, `other`.
   - Allowed priorities: `critical`, `high`, `normal`, `low`.
   - Allowed urgencies: `immediate`, `high`, `normal`, `low`.
   - ISO 8601 deadline validation or `null`.
   - Confidence bounds `0.0` to `1.0`.

4. **Deterministic Decision Engine**:
   - Applies clear business logic independently of the LLM:
     - `college` → `AI/COLLEGE`
     - `college` + `high`/`critical` priority → `AI/COLLEGE/HIGH_PRIORITY`
     - `action_required == true` → `AI/ACTION_REQUIRED`
     - `urgency == immediate` or `priority == critical` → Star email
     - `newsletter` → `AI/NEWSLETTER`
     - `promotion` → `AI/PROMOTION`
     - `spam` → `AI/SPAM_REVIEW`
   - Zero destructive actions (no deletes, no automated email sending).

5. **State Tracking & Idempotency**:
   - Complete state machine: `DISCOVERED` → `PARSED` → `CLASSIFIED` → `VALIDATED` → `ACTIONED` → `COMPLETE`.
   - SQLite persistence ensures processed emails are never processed again, even across restarts.
   - Failures mark the email `FAILED` with an error message and allow safe retry without crashing the worker.

6. **Safety & Security**:
   - Default `DRY_RUN=true` to test and preview actions without modifying your inbox.
   - Default `INITIAL_LOOKBACK_HOURS=0` prevents accidental processing of your entire inbox history on first launch.
   - No sensitive email contents are logged (only IDs, subjects truncated, categories, and action summaries).
   - Trace IDs correlate logs for each email processed.

---

## Installation

### Prerequisites
- Python 3.11 or higher
- A Google Cloud Project with the **Gmail API** enabled

### Setup Steps

1. Clone or navigate to the repository directory:
   ```bash
   cd mail_sorter
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create your local configuration:
   ```bash
   copy .env.example .env
   # On Linux/macOS:
   cp .env.example .env
   ```

---

## Gmail OAuth2 Configuration

1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g. `MailSorter-Automation`).
3. Navigate to **APIs & Services → Library**, search for **Gmail API**, and click **Enable**.
4. Go to **APIs & Services → OAuth consent screen**:
   - Select **External** (or Internal for Google Workspace).
   - Fill in the required app details.
   - Add scopes:
     - `https://www.googleapis.com/auth/gmail.modify`
     - `https://www.googleapis.com/auth/gmail.labels`
   - Add your Gmail address as a **Test User**.
5. Go to **APIs & Services → Credentials**:
   - Click **Create Credentials → OAuth client ID**.
   - Choose **Desktop app**.
   - Download the client secrets JSON and save it as `credentials.json` in the `mail_sorter` root directory.
6. On the first run, the app will launch a browser window requesting authorization. After consent is given, an OAuth token is saved to `token.json` for subsequent unattended runs.

---

## AI / Classifier Configuration

The application supports three classification providers through the `BaseClassifier` abstraction:

### Option A: 100% Offline Rule Engine (Default — Zero Cloud, Zero Keys, Instant)
No accounts, no downloads, and no internet required for AI. Runs entirely on your CPU:
```ini
LLM_PROVIDER=offline
```

### Option B: Local Ollama LLM (100% Private, Local Model)
Run local models like `llama3.2` or `mistral` on your own hardware via Ollama:
```ini
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

### Option C: Google Gemini (Free Cloud Tier)
Use Google's free Gemini API:
```ini
LLM_PROVIDER=google
LLM_MODEL=gemini-2.0-flash
LLM_API_KEY=your-free-gemini-api-key
```

---

## Running the Application

### 1. Dry-Run Mode (Recommended First Run)

By default, `DRY_RUN=true` in `.env.example`. In dry-run mode, emails are fetched, parsed, cleaned, classified, and actions are determined, but **no changes are made to Gmail**.

```bash
python main.py
```

Console output will display structured dry-run reports:
```
============================================================
DRY-RUN REPORT
============================================================
Subject   : Assignment 3 deadline changed
Category  : college
Priority  : high
Urgency   : high
Action?   : Yes
Deadline  : 2025-10-31T23:59:00
Confidence: 0.95
Summary   : Professor Smith announced Assignment 3 is due this Friday.
------------------------------------------------------------
ACTIONS:
  1. add_label [AI/COLLEGE] — Category is college
  2. add_label [AI/COLLEGE/HIGH_PRIORITY] — College email with high/critical priority
  3. add_label [AI/ACTION_REQUIRED] — Email requires user action
  4. star — Urgency is immediate / high priority
============================================================
```

### 2. Live Execution Mode

Once satisfied with dry-run classifications, enable live actions in `.env`:
```ini
DRY_RUN=false
```
Then run:
```bash
python main.py
```
The application will:
- Check that all `AI/*` labels exist in your Gmail account (and create them automatically if missing).
- Poll for new unread emails every `POLL_INTERVAL_SECONDS` (default: 300s / 5 minutes).
- Apply labels, stars, and record the results in SQLite.

---

## Running the Test Suite

The test suite includes 135+ tests with full coverage across all subsystems. Unit tests use mocks and fixtures, requiring no network connection, Gmail credentials, or LLM API keys.

Run all tests:
```bash
pytest tests/ -v
```

Run tests with test coverage reporting:
```bash
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Extensibility & Future Work

The system is designed with explicit interfaces allowing straightforward extensions:
- **New LLM Providers**: Subclass `BaseClassifier` in `ai/classifier.py` (e.g. OpenAI, Anthropic, or Ollama local models) without changing the decision engine or reader.
- **Custom Decision Rules**: Extend `decide()` in `engine/decision_engine.py` with custom rules (e.g. VIP sender lists, automated folder archives).
- **Notification Services**: Add event hooks in `_execute_actions` in `main.py` for Slack, Discord, desktop notifications, or Calendar events.
- **Attachment Inspection**: Extend `processing/attachments.py` to extract text from PDF or DOCX files for classification.
