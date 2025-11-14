# Financial Chatbot - DistilBERT + LLaMA + RAG

A production-ready financial chatbot built with Flask, featuring hybrid AI architecture combining DistilBERT for intent detection, LLaMA for natural language generation, and FAISS-based RAG for document retrieval.

---

## Quick Start

### Prerequisites
- **Python**: 3.10+ (3.13 installed)
- **RAM**: 16GB+ system memory
- **OS**: Windows 10/11 or Linux
- **Database**: Supabase (cloud PostgreSQL)

### Installation (5 minutes)

```bash
# 1. Clone Repository
git clone <https://github.com/busilas/MScCP/tree/main/application>
cd application

# 2. Navigate to project
cd project

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python main.py init_db
python main.py create_admin

# 5. Install Ollama (for LLaMA)
# Download from: https://ollama.com/download
ollama pull llama3.2:3b

# 6. Run application
python main.py
# Access at: http://localhost:5000
```

### Default Credentials
- **Username**: `admin`
- **Password**: `admin123`

---

## Features

### Core Functionality
- **Hybrid AI Architecture**: DistilBERT + LLaMA 3.2 + FAISS RAG
- **3 LLaMA Options**: Groq API (free), Ollama (local), or OpenAI
- **Advanced Prompt Engineering**: Security-first prompts with compliance disclaimers
- **Few-Shot Learning**: 5 financial examples for consistent responses
- **Mode Switching**: DistilBERT-only, LLaMA-only, or Hybrid modes
- **Guardian PII Protection**: Automatic detection and redaction of sensitive information
- **Document Upload**: Support for PDF, TXT, CSV with automatic FAISS indexing
- **Real-time Chat Interface**: Responsive Bootstrap UI

### Admin Features
- **Settings Dashboard**: Toggle authentication, switch chatbot modes
- **Performance Analytics**: Track latency, mode usage, Guardian detections
- **User Management**: Role-based access control
- **Training Data Management**: Upload and preprocess training datasets

### Security
- **Flask-Login Authentication**: Secure user sessions with bcrypt
- **PII Detection**: Regex + ML-based Guardian plugin
- **Row-Level Security**: PostgreSQL with proper authorization
- **Input Validation**: File type, size, and MIME type checking

---

## Architecture

```
          ┌─────────────────┐
          │   User Input    │       
          └────────┬────────┘
                   │
              ┌────▼────┐
              │Guardian │ (PII Detection)
              └────┬────┘
                   │
    ┌──────────────▼────────────────┐
    │    Intent Classification      │
    │      (DistilBERT)             │
    │  + Entity Extraction          │
    └─────────────┬─────────────────┘
                  │
    ┌─────────────▼─────────────────┐
    │   Document Retrieval (RAG)    │
    │      (FAISS + Embeddings)     │
    └─────────────┬─────────────────┘
                  │
    ┌─────────────▼─────────────────┐
    │   Prompt Engineering          │
    │  • Security guidelines        │
    │  • Compliance disclaimers     │
    │  • Few-shot examples          │
    │  • Chain-of-thought           │
    │  • Intent-based guidance      │
    └─────────────┬─────────────────┘
                  │
    ┌─────────────▼─────────────────┐
    │  Response Generation          │
    │  (LLaMA via Groq/Ollama/API)  │
    └─────────────┬─────────────────┘
                  │
             ┌────▼────┐
             │Guardian │ (Output Filtering)
             └────┬────┘
                  │
           ┌──────▼──────┐
           │   Response  │
           └─────────────┘
```

---

## Configuration

### LLaMA Options (Choose One)

The chatbot automatically tries all configured LLaMA providers in priority order:

#### Option 1: Groq API (Recommended - Free & Fast)

**Setup:**
```bash
# 1. Get free API key from: https://console.groq.com/keys
# 2. Add to .env:
GROQ_API_KEY=gsk_your_key_here
# 3. Restart Flask
```

**Pros:** Free, very fast (10-20 tokens/sec), no installation
**Cons:** Requires internet, rate limits apply

---

#### Option 2: Ollama (Best for Privacy)

**Setup:**
```bash
# 1. Download from: https://ollama.com/download
# 2. Install and run Ollama
# 3. Pull model:
ollama pull llama3.2:3b
# 4. Configure .env (already set):
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
# 5. Restart Flask
```

**Pros:** Free, private (offline), no API limits
**Cons:** Uses 2GB RAM, slower than Groq (5-10 tokens/sec)

---

#### Option 3: OpenAI (Paid)

**Setup:**
```bash
# 1. Get API key from: https://platform.openai.com/api-keys
# 2. Add to .env:
OPENAI_API_KEY=sk-your_key_here
# 3. Restart Flask
```

**Pros:** Best quality responses, very reliable
**Cons:** Costs ~$0.002 per query

---

### Environment Variables

**Your `.env` file:**

```env
# Supabase Database (Configured)
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_SUPABASE_ANON_KEY=your-anon-key

# Flask Configuration
SECRET_KEY=auto-generated-or-set-your-own
FLASK_ENV=development

# DistilBERT Configuration (Local)
USE_LOCAL_DISTILBERT=True
DISTILBERT_MODEL_NAME=distilbert-base-uncased

# LLaMA Configuration
USE_LOCAL_LLAMA=False
LLAMA_MODEL_NAME=google/flan-t5-base

# Ollama (Local LLaMA)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Groq API (Free LLaMA)
GROQ_API_KEY=your-groq-key

# OpenAI API (Paid)
OPENAI_API_KEY=your-openai-key

# Hugging Face API (Legacy - Optional)
HUGGINGFACE_API_KEY=your-hf-key

# File Upload Settings
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=pdf,txt,csv

# FAISS Configuration
FAISS_INDEX_PATH=data/faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Usage

### 1. Chat Interface

**Access:** http://localhost:5000

**Example Queries:**
```
"What is compound interest?"
"How do I apply for a mortgage?"
"Explain the difference between a Roth IRA and Traditional IRA"
"What are the tax implications of selling stocks?"
```

### 2. Admin Dashboard

**Access:** http://localhost:5000/admin/

**Features:**
- Change chatbot mode (distilbert, llama, hybrid)
- Toggle authentication
- View performance metrics
- Manage users

### 3. Document Upload

**Access:** http://localhost:5000/upload

**Steps:**
1. Drag and drop or select files (PDF, TXT, CSV)
2. Files are automatically:
   - Validated for type and size
   - Text extracted
   - Embedded and indexed in FAISS
3. Chatbot uses uploaded documents for context

### 4. Training Interface

**Access:** http://localhost:5000/training

**Steps:**
1. Upload Q&A dataset (JSON format)
2. Click "Train Model"
3. System fine-tunes DistilBERT on your data

**Training Data Format (JSON):**
```json
[
  {
    "question": "What is a stock?",
    "answer": "A stock represents ownership in a company..."
  },
  {
    "question": "How does compound interest work?",
    "answer": "Compound interest is interest calculated on..."
  }
]
```

---

## Chatbot Modes

### DistilBERT Mode
- Intent classification only
- Entity extraction
- Lightweight and fast (50-100ms response)
- Best for: Simple queries, pattern matching

### LLaMA Mode
- Natural language generation only
- No intent analysis
- Higher quality responses (1-3s response)
- Best for: Complex questions, conversational responses

### Hybrid Mode (Recommended)
- Complete pipeline
- Intent detection → RAG retrieval → LLaMA generation
- Best performance and accuracy
- Best for: Production use, complex financial queries

**Change mode:** Admin Dashboard → Settings → Chatbot Mode

---

## API Reference

### Chat API

**POST /api/query**

**Request:**
```json
{
  "query": "What is the stock market?"
}
```

**Response:**
```json
{
  "response": "The stock market is a platform where shares...",
  "mode_used": "hybrid",
  "latency_ms": 1250,
  "guardian_flag": false,
  "intent": "market_data",
  "similarity_score": 0.8543,
  "model": "ollama/llama3.2:3b"
}
```

### Admin API

**POST /admin/settings**
```json
{
  "auth_required": true,
  "chatbot_mode": "hybrid",
  "use_local_distilbert": true
}
```

**GET /admin/api/stats**
```json
{
  "performance": {
    "total_queries": 1250,
    "avg_latency_ms": 1100,
    "guardian_redaction_rate": 2.5
  }
}
```

---

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app tests/

# Specific test suite
pytest tests/test_guardian.py
pytest tests/test_api.py
pytest tests/test_auth.py
pytest tests/test_prompt_engineering.py
pytest tests/test_adversarial_attacks.py
```

### Test Categories

1. **Unit Tests**
   - Guardian PII detection
   - Database models
   - Data preprocessing

2. **Integration Tests**
   - API endpoints
   - Authentication flows
   - File upload validation

3. **Security Tests**
   - Adversarial attacks
   - Prompt injection attempts
   - PII leakage prevention

---

## Training

### Train DistilBERT (Intent Classification)

```bash
# Via Web Interface (Recommended)
1. Navigate to: http://localhost:5000/training
2. Upload training data (JSON format)
3. Click "Train Model"

# Via Command Line
python utils/train_distilbert.py
```

**Training Time:** 5-10 minutes for 100 samples

### Fine-tune LLaMA (Not Required)

LLaMA works out-of-the-box via Ollama/Groq. Fine-tuning is optional for specialized financial terminology:

```bash
python utils/train_llama.py \
  --data training/examples/sample_qa.json \
  --epochs 3 \
  --batch-size 4 \
  --output models/llama-finetuned
```

**Note:** Requires NVIDIA GPU with 10GB+ VRAM for local fine-tuning.

---


## Troubleshooting

### Common Issues

**1. Ollama Connection Error**

```
ℹ Ollama not available (install from https://ollama.com)
```

**Solution:**
```bash
# Start Ollama service
ollama serve

# Verify model is pulled
ollama list
# Should show: llama3.2:3b
```

---

**2. Database Connection Error**

```
Could not connect to database
```

**Solution:**
```bash
# Check .env has correct Supabase credentials
# Verify at: https://app.supabase.com/project/<your-project>/settings/api

# Test connection
python -c "from app import create_app; app = create_app(); print('DB connected!')"
```

---

**3. FAISS Index Not Found**

```
FileNotFoundError: data/faiss_index
```

**Solution:**
```bash
# Upload documents via web UI to auto-build index
# OR rebuild manually:
python utils/faiss_builder.py training/clean/data.txt data/faiss_index
```

---

**4. Guardian False Positives**

**Solution:**
```python
# Adjust patterns in app/services/guardian.py
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    # Modify as needed
}
```

---

**5. Out of Memory (Training)**

**Solution:**
```bash
# Reduce batch size in training config
# Edit utils/train_distilbert.py:
BATCH_SIZE = 4  # Instead of 8
```

---

## Project Structure

```
project/
├── app/
│   ├── __init__.py                    # Flask app factory
│   ├── config.py                      # Configuration management
│   ├── models/                        # Database models (SQLAlchemy)
│   │   ├── user.py                    # User authentication
│   │   ├── document.py                # Document metadata
│   │   ├── log.py                     # Query logs
│   │   ├── settings.py                # System settings
│   │   ├── training_file.py           # Training data tracking
│   │   └── training_job.py            # Training job status
│   ├── routes/                        # Flask blueprints
│   │   ├── auth.py                    # Login/register/logout
│   │   ├── chat.py                    # Chatbot query endpoint
│   │   ├── upload.py                  # Document upload & FAISS
│   │   ├── training.py                # Model training interface
│   │   └── admin.py                   # Admin dashboard
│   ├── services/                      # AI/ML services
│   │   ├── distilbert_service.py      # Intent classification
│   │   ├── llama_service.py           # Text generation (LLaMA 3.2)
│   │   ├── rag_service.py             # FAISS document retrieval
│   │   ├── guardian.py                # PII detection/redaction
│   │   ├── prompt_engineering.py      # Advanced prompt templates
│   │   ├── chatbot_service.py         # Orchestration layer
│   │   └── training_service.py        # Fine-tuning pipeline
│   └── templates/                     # HTML templates
│       ├── base.html                  # Base layout
│       ├── chat.html                  # Chat interface
│       ├── upload.html                # Document upload
│       ├── training.html              # Training dashboard
│       ├── admin.html                 # Admin panel
│       └── performance.html           # Analytics
├── tests/                             # Test suites (pytest)
│   ├── test_api.py                    # API endpoint tests
│   ├── test_auth.py                   # Authentication tests
│   ├── test_guardian.py               # PII detection tests
│   ├── test_prompt_engineering.py     # Prompt template tests
│   ├── test_adversarial_attacks.py    # Security tests
│   └── adversarial_attacks_dataset.json
├── training/                          # Training data storage
│   ├── raw/                           # Unprocessed documents
│   ├── clean/                         # Preprocessed data
│   └── examples/
│       └── sample_qa.json             # Sample Q&A dataset
├── utils/                             # Utility scripts
│   ├── preprocess.py                  # Data preprocessing
│   ├── train_llama.py                 # LLaMA fine-tuning (LoRA)
│   ├── train_distilbert.py            # DistilBERT training
│   ├── synthetic_queries_generator_fixed.py
│   ├── forum_queries_generator_fixed.py
│   └── phrasebank_queries_generator_fixed.py
├── uploads/                           # User-uploaded documents
├── data/
│   └── faiss_index/                   # FAISS vector index
├── models/                            # Saved fine-tuned models
├── main.py                            # Application entry point
├── requirements.txt                   # Python dependencies
├── pytest.ini                         # Test configuration
├── .env                               # Environment variables
├── .env.example                       # Template
└── README.md                          # This file
```

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (Intel/AMD)
- **RAM**: 8GB
- **Storage**: 5GB free space
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10+

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 10GB+ SSD
- **OS**: Windows 11 or Ubuntu 22.04
- **Python**: 3.12+
- **GPU**: Optional (NVIDIA with CUDA for local LLaMA fine-tuning)

---

## Performance

### Response Times (Typical)

| Mode | Latency | Quality |
|------|---------|---------|
| DistilBERT Only | 50-100ms | Basic pattern matching |
| LLaMA Only (Groq) | 500-1000ms | High-quality conversational |
| LLaMA Only (Ollama) | 1000-3000ms | High-quality conversational |
| Hybrid | 1000-3000ms | Best (intent + context + generation) |

### Resource Usage

| Component | CPU | RAM | Notes |
|-----------|-----|-----|-------|
| Flask App | 5-10% | 200-500MB | Base application |
| DistilBERT | 20-30% | 1-2GB | During inference |
| Ollama (LLaMA 3.2) | 30-50% | 2-3GB | Local model |
| FAISS Index | 5% | 100-500MB | Depends on document count |

---

## Academic Context

This project demonstrates:

1. **Hybrid AI Architecture**: Combining specialized models for optimal performance
2. **RAG Implementation**: Document-grounded response generation
3. **Privacy Protection**: Guardian plugin for PII detection
4. **Flexible LLaMA Deployment**: API-first with local fallback options
5. **Production Patterns**: Authentication, logging, monitoring

### Key Features for MSc Demonstration

- Multi-model orchestration (DistilBERT + LLaMA + RAG)
- Advanced prompt engineering techniques
- Security-first design (PII detection, input validation)
- Multiple LLaMA deployment options (cloud API, local, hybrid)
- Comprehensive testing suite
- Production-ready code structure

### Limitations

- Small training dataset (proof-of-concept)
- LoRA fine-tuning only (not full LLaMA training)
- Single-language support (English)
- Basic RAG implementation (no reranking)

### Future Enhancements

- Multi-language support
- Voice interface
- Real-time market data integration
- Advanced RAG with reranking
- Federated learning for privacy
- Model distillation for efficiency
- Mobile application

---

## License

Academic project for MSc demonstration purposes.

---

## Acknowledgments

- **Hugging Face Transformers** - DistilBERT and model infrastructure
- **Meta LLaMA** - Large language model
- **Ollama** - Local LLaMA deployment
- **Groq** - Fast LLaMA inference API
- **FAISS** - Vector similarity search by Facebook Research
- **Flask** - Web framework
- **Supabase** - PostgreSQL database
- **Bootstrap** - UI components

---

## Quick Reference

### Start Application
```bash
python main.py
```

### Access Points
- **Chat**: http://localhost:5000
- **Admin**: http://localhost:5000/admin/
- **Upload**: http://localhost:5000/upload
- **Training**: http://localhost:5000/training

### Change LLaMA Provider Priority
Edit `app/services/llama_service.py` line 269-285 to reorder:
1. Groq (free, fast)
2. OpenAI (paid, best quality)
3. Ollama (local, private)
4. Hugging Face (legacy)

### Default Admin Credentials
- **Username**: `admin`
- **Password**: `admin123`

**Change immediately after first login!**

