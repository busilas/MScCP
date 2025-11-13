# Financial Chatbot Query Generators

Three Python scripts for generating synthetic queries for financial chatbot testing, based on Section 3.3.4: Data Pre-processing.

## Overview

This suite generates **900 total queries** across three datasets:

| Generator      | Queries | Purpose                                             |
|----------------|---------|-----------------------------------------------------|
| **Synthetic**  | 625     | Faker-generated with ε-differential privacy (ε=1.0) |
| **Forum**      | 100     | Real-world anonymized forum queries                 |
| **PhraseBank** | 175     | NordicBank corpus with 20% overlap validation       |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-generators.txt
```

Or manually:

```bash
pip install pandas numpy Faker pyarrow presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg
```

### 2. Run Generators

```bash
# Generate all datasets
python synthetic_queries_generator_fixed.py
python forum_queries_generator_fixed.py
python phrasebank_queries_generator_fixed.py
```

### 3. Output Files

Each generator creates three files with timestamp:

- `{type}_queries_{timestamp}.csv` - CSV format
- `{type}_queries_{timestamp}.json` - JSON with metadata
- `{type}_queries_{timestamp}.parquet` - Parquet format (optimized)

## Generator Details

### 1. Synthetic Queries Generator (625 queries)

**Distribution:**
- 40% PII-heavy queries (250) - Account numbers, IBANs, emails, phone numbers
- 30% Compliance queries (188) - GDPR, MiFID II, data protection
- 30% General banking (187) - Interest rates, loans, account management

**Features:**
- Uses Python Faker library with multi-locale support (US, GB, SE, FI, NO)
- Implements ε-differential privacy (ε=1.0) with Laplace noise
- Generates 350 Faker queries + 275 curated templates
- Average query length: 512 tokens (matches RAG chunk size)

**Privacy Protection:**
- Laplace noise added to monetary amounts
- Sensitivity parameter: 100 for currency values
- All generated PII is synthetic and non-functional

**Example Queries:**
```
PII-heavy: "What is the balance on my account SE7842341234567?"
Compliance: "How does your bank comply with GDPR Article 15 regarding data access?"
General: "What are your current interest rates for savings accounts?"
```

### 2. Forum Queries Generator (100 queries)

**Source:** Real-world banking forum discussions (anonymized)

**Features:**
- Multi-stage anonymization using Microsoft Presidio
- PII detection and replacement with semantic placeholders
- Preserves query structure and context
- Nordic-specific patterns (Swedish, Finnish, Norwegian, Danish)

**Anonymization Pipeline:**
1. PII Detection (Presidio + custom patterns)
2. Nordic-specific entity recognition
3. Structural preservation for validation
4. Semantic placeholder generation

**Detected PII Types:**
- IBAN with check-digit preservation
- Account numbers with format retention
- Email addresses (bank domain awareness)
- Nordic phone numbers (+46, +358, +47, +45)
- Names and addresses
- National IDs (personnummer, henkilötunnus, fødselsnummer)
- Currency amounts

**Example Queries:**
```
Original: "My IBAN SE4550000000058398257466 shows wrong balance"
Anonymized: "My IBAN [REDACTED_IBAN_1001] shows wrong balance"

Original: "Call me at +46701234567 about my loan"
Anonymized: "Call me at +46-555-7823 about my loan"
```

### 3. PhraseBank Queries Generator (175 queries)

**Source:** NordicBank PhraseBank corpus

**Strategy:**
- 20% overlap queries (35) - For consistency validation with synthetic data
- 80% edge cases (140) - Complex scenarios and boundary testing

**Category Distribution:**
| Category                 | Count | %     |
|--------------------------|-------|-------|
| Consistency validation   | 35    | 20%   |
| Multi-currency           | 20    | 11.4% |
| Regulatory compliance    | 25    | 14.3% |
| Complex transactions     | 25    | 14.3% |
| Authentication/security  | 20    | 11.4% |
| Account lifecycle        | 15    | 8.6%  |
| Fee disputes             | 15    | 8.6%  |
| Data/reporting           | 10    | 5.7%  |
| Nordic-specific          | 10    | 5.7%  |

**Overlap Purpose:**
The 20% overlap with synthetic queries validates:
- Consistency across different data sources
- Cross-dataset response quality
- Model robustness to query variations

**Edge Case Focus (80%):**
- Multi-currency conversion edge cases
- GDPR compliance boundary scenarios
- Complex transaction disputes
- Authentication failures and fallbacks
- Account lifecycle edge conditions
- Fee calculation disputes
- Nordic banking regulations

**Example Edge Cases:**
```
Multi-currency: "Exchange rate changed between initiating transfer and execution.
Which rate applies and who bears the difference?"

Regulatory: "Under GDPR Article 22, do you use automated decision-making for
loan applications? Can I request human review?"

Nordic-specific: "BankID authentication mandatory for Swedish tax filing but
I don't have BankID. Alternative for Nordic residents abroad?"
```

## Output Schema

All generators produce consistent schema:

```json
{
  "query_id": "SYN_0001",
  "query_text": "What is the balance on my account...",
  "category": "PII-heavy",
  "source": "faker_generated",
  "pii_entities": ["IBAN", "EMAIL"],
  "token_count": 512,
  "complexity": "high",
  "differential_privacy": "epsilon_1.0"
}
```

## Usage in Testing

### 1. RAG System Testing
- Query lengths match 512-token chunk size
- Tests retrieval accuracy across query types
- Validates context window handling

### 2. Privacy Validation
- PII detection and masking
- Differential privacy guarantees
- Anonymization effectiveness

### 3. Compliance Testing
- GDPR Article coverage
- MiFID II requirements
- EU AI Act transparency

### 4. Edge Case Validation
- Boundary conditions
- Error handling
- Complex scenario resolution

### 5. Consistency Validation
- Cross-dataset comparison (20% overlap)
- Response quality metrics
- Model robustness assessment

## Dataset Statistics

### Combined Dataset (900 queries)

**Category Distribution:**
- PII-heavy: 250 (27.8%)
- Compliance: 238 (26.4%)
- General banking: 187 (20.8%)
- Edge cases: 140 (15.6%)
- Forum queries: 100 (11.1%)

**Token Distribution:**
- Average: 512 tokens
- Range: 50-800 tokens
- Optimized for RAG chunk alignment

**Privacy Features:**
- 350 queries with ε-DP (ε=1.0)
- 275 queries with Presidio anonymization
- 100% PII coverage in testing data

## Technical Details

### Differential Privacy (ε=1.0)

```python
scale = sensitivity / epsilon  # 100 / 1.0 = 100
noise = np.random.laplace(0, scale)
noisy_amount = original_amount + noise
```

**Privacy Guarantee:**
- ε=1.0 provides strong privacy protection
- Laplace mechanism for monetary values
- Sensitivity=100 for currency amounts

### Anonymization Stages

1. **Detection:** Presidio NER + custom regex patterns
2. **Validation:** Nordic-specific format checking
3. **Replacement:** Semantic placeholders with structure preservation
4. **Verification:** PII leakage detection

### Nordic Banking Specifics

**Supported Countries:**
- 🇸🇪 Sweden: Personnummer, BankID, SEK, +46
- 🇫🇮 Finland: Henkilötunnus, FI-IBAN, EUR, +358
- 🇳🇴 Norway: Fødselsnummer, NOK, +47
- 🇩🇰 Denmark: CPR, DKK, +45

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- Faker >= 18.0.0
- pyarrow >= 12.0.0
- presidio-analyzer >= 2.2.0
- presidio-anonymizer >= 2.2.0
- spacy >= 3.5.0
- en_core_web_lg model

## Troubleshooting

### ModuleNotFoundError
```bash
pip install --upgrade -r requirements-generators.txt
```

### Spacy Model Missing
```bash
python -m spacy download en_core_web_lg
```

### Memory Issues (Large Datasets)
```python
# Process in chunks
df.to_parquet(filename, engine='pyarrow', compression='snappy')
```

### Presidio Installation Issues
```bash
# Install with all extras
pip install presidio-analyzer[all]
pip install presidio-anonymizer[all]
```

## Data Safety

⚠️ **Important:**
- All generated PII is synthetic and non-functional
- Real customer data is never used
- Forum queries are anonymized using Presidio
- IBANs are structurally valid but not real accounts
- Phone numbers use reserved ranges

## Citation

Based on dissertation work:
- Section 3.3.4: Data Pre-processing
- NordicBank PhraseBank corpus
- ε-differential privacy implementation
- Microsoft Presidio anonymization framework

## License

Research and educational use only.

## Support

For issues or questions about the generators:
1. Check requirements are correctly installed
2. Verify Python version (3.8+)
3. Ensure spacy model is downloaded
4. Review output logs for specific errors

## Output Example

```
======================================================================
Synthetic Queries Generator for Financial Chatbot
Based on Dissertation Section 3.3.4: Data Pre-processing
======================================================================

Generating synthetic queries...
✓ Generated 625 queries
  - PII-heavy: 250
  - Compliance: 188
  - General banking: 187

✓ Saved CSV: synthetic_queries_20251109_143052.csv
✓ Saved JSON: synthetic_queries_20251109_143052.json
✓ Saved Parquet: synthetic_queries_20251109_143052.parquet

=== Dataset Summary ===
Total queries: 625

Category distribution:
PII-heavy           250
compliance          188
general_banking     187

Source distribution:
faker_generated     350
curated            275

Average token count: 512.3
Token count range: 52 - 798

======================================================================
Generation complete!
======================================================================
```

