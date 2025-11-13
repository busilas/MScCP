# Financial Chatbot - DistilBERT + LLaMA + RAG

A production-ready financial chatbot built with Flask, featuring hybrid AI architecture combining DistilBERT for intent detection, LLaMA for natural language generation, and FAISS-based RAG for document retrieval.

## Features

### Core Functionality
- **Hybrid AI Architecture**: DistilBERT + LLaMA + FAISS RAG
- **Advanced Prompt Engineering**: Security-first prompts with compliance disclaimers
- **Few-Shot Learning**: 5 financial examples for consistent responses
- **Mode Switching**: DistilBERT-only, LLaMA-only, or Hybrid modes
- **Local/API Flexibility**: Run models locally or via HuggingFace API
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
    └──────────────┬────────────────┘
                   │
    ┌──────────────▼────────────────┐
    │   Document Retrieval (RAG)    │
    │      (FAISS + Embeddings)     │
    └──────────────┬────────────────┘
                   │
    ┌──────────────▼────────────────┐
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
    │      (LLaMA)                  │
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

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 15+ (or use Supabase)
- 16GB+ RAM (for local LLaMA)
- GPU recommended (for training)

### Quick Start

1. **Clone Repository**
```bash
git clone <https://github.com/busilas/MScCP/tree/main/application>
cd application
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Initialize Database**
```bash
python main.py init_db
python main.py create_admin
```

5. **Initialize Default User Accounts**
Two user accounts are automatically created:

### Admin Account
- **Username:** `admin`
- **Password:** `admin123`
- **Email:** admin@example.com
- **Permissions:** Full system access

### Regular User Account
- **Username:** `user`
- **Password:** `user123`
- **Email:** user@example.com
- **Permissions:** Chat and document upload

These are created by running:
```bash
python main.py create_default_users
```

6. **Run Application**
```bash
python main.py
# Access at http://localhost:5000
```

## Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/financial_chatbot

# Flask
SECRET_KEY=your-secret-key
FLASK_ENV=development

# Model Configuration
USE_LOCAL_DISTILBERT=True
USE_LOCAL_LLAMA=True

# API Keys (if not using local models)
HUGGINGFACE_API_KEY=your-key
LLAMA_API_KEY=your-key

# File Upload
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=pdf,txt,csv

# FAISS
FAISS_INDEX_PATH=data/faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Model Modes

**DistilBERT Mode**
- Intent classification only
- Entity extraction
- Lightweight and fast

**LLaMA Mode**
- Natural language generation
- No intent analysis
- Higher quality responses

**Hybrid Mode** (Recommended)
- Complete pipeline
- Intent detection → RAG retrieval → LLaMA generation
- Best performance

## Document Management

### Upload Documents

1. Navigate to **Upload** page
2. Drag and drop or select files (PDF, TXT, CSV)
3. Files are automatically:
   - Validated for type and size
   - Text extracted
   - Embedded and indexed in FAISS

### Rebuild FAISS Index

```bash
# Via Admin UI
Admin Dashboard → Upload → Rebuild Index

# Via API
POST /upload/api/rebuild-index
```

## Training

### Prepare Training Data

```bash
# Preprocess raw text
python utils/data_preparation.py training/raw/file.txt training/clean/file.txt

# Build FAISS index
python utils/faiss_builder.py training/clean/file.txt data/faiss_index
```

### Fine-tune LLaMA (Proof-of-Concept)

```bash
python utils/train_llama.py \
  --data training/examples/sample_qa.json \
  --epochs 3 \
  --batch-size 4 \
  --output models/llama-finetuned
```

**Note**: This demonstrates small-dataset training on limited hardware. Full-scale training requires substantial compute resources.

### Training Data Format

**Q&A Pairs (JSON)**
```json
[
  {
    "question": "What is a stock?",
    "answer": "A stock represents ownership..."
  }
]
```

**Raw Text (TXT)**
```
Financial Markets Overview
The stock market is a complex ecosystem...
```

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app tests/

# Specific test suite
pytest tests/test_guardian.py
pytest tests/test_models.py
pytest tests/test_routes.py
pytest tests/test_api.py
pytest tests/test_auth.py
python tests/test_adversarial_attacks.py
python tests/test_prompt_engineering.py
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

3. **End-to-End Tests**
   - Complete user journeys
   - Mode switching
   - Performance logging

## API Reference

### Chat API

**POST /api/query**
```json
{
  "query": "What is the stock market?"
}
```

Response:
```json
{
  "response": "The stock market is...",
  "mode_used": "hybrid",
  "latency_ms": 1250,
  "guardian_flag": false,
  "intent": "market_data",
  "similarity_score": 0.8543
}
```

### Admin API

**POST /admin/settings**
```json
{
  "auth_required": true,
  "chatbot_mode": "hybrid",
  "use_local_distilbert": true,
  "use_local_llama": true
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

## Performance Optimization

### Local Models
- Use 8-bit quantization for LLaMA
- Batch FAISS queries
- Cache embeddings
- Enable GPU acceleration

### API Mode
- Implement request queuing
- Add rate limiting
- Cache frequent queries
- Monitor API costs

## Troubleshooting

### Common Issues

**1. Out of Memory (LLaMA)**
```bash
# Use smaller model or 8-bit quantization
USE_LOCAL_LLAMA=False  # Switch to API mode
```

**2. FAISS Index Not Found**
```bash
# Rebuild index
python utils/faiss_builder.py training/clean/data.txt data/faiss_index
```

**3. Database Connection Error**
```bash
# Check PostgreSQL is running
sudo service postgresql status

# Test connection
psql -h localhost -U postgres -d financial_chatbot
```

**4. Guardian False Positives**
```python
# Adjust regex patterns in app/services/guardian.py
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    # Modify patterns as needed
}
```

## Project Structure

```
application/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── models/           # Database models
│   ├── routes/           # Flask blueprints
│   ├── services/         # AI services (DistilBERT, LLaMA, RAG, Guardian)
│   └── templates/        # HTML templates
├── tests/                # Test suites
├── training/             # Training data
│   ├── raw/
│   ├── clean/
│   └── examples/
├── utils/                # Utility scripts
├── generators/           # Synthetic queries scripts
├── main.py               # Application entry point
├── requirements.txt
├── Dockerfile
└── README.md
```

## Academic Context

This project demonstrates:

1. **Hybrid AI Architecture**: Combining specialized models for optimal performance
2. **RAG Implementation**: Document-grounded response generation
3. **Privacy Protection**: Guardian plugin for PII detection
4. **Small-Dataset Training**: LoRA fine-tuning on limited resources
5. **Production Patterns**: Authentication, logging, monitoring

### Limitations

- Small training dataset (proof-of-concept)
- Limited hardware constraints (MSc project)
- LoRA fine-tuning only (not full training)
- API fallbacks for resource-intensive models

### Future Enhancements

- Multi-language support
- Voice interface
- Real-time market data integration
- Advanced RAG with reranking
- Federated learning for privacy
- Model distillation for efficiency

## License

Academic project for MSc in Computer Science demonstration purposes.

## Support

For issues or questions:
- Check troubleshooting section
- Review test cases for examples
- Examine code comments and docstrings

## Acknowledgments

- HuggingFace Transformers
- Meta LLaMA
- FAISS by Facebook Research
- Flask framework
- Bootstrap UI components
