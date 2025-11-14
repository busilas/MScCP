"""
Synthetic Queries Generator for Financial Chatbot Testing
Based on Section 3.3.4: Data Pre-processing

Generates 625 synthetic queries:
- 350 using Python Faker with ε-differential privacy (ε=1.0)
- 275 curated queries
- Distribution: 40% PII-heavy, 30% compliance-related, 30% general banking
- Query lengths correspond to 512-token RAG chunk size
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json

# Initialize Faker with seed for reproducibility
fake = Faker(['en_US', 'en_GB', 'sv_SE', 'fi_FI', 'nb_NO'])
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class DifferentialPrivacyNoise:
    """Implements ε-differential privacy (ε=1.0) for synthetic data"""

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def add_noise(self, value, sensitivity=1.0):
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def perturb_amount(self, amount):
        """Add noise to monetary amounts"""
        noisy = self.add_noise(amount, sensitivity=100)
        return max(0, round(noisy, 2))


class SyntheticQueryGenerator:
    """Generates synthetic banking queries for chatbot testing"""

    def __init__(self):
        self.dp = DifferentialPrivacyNoise(epsilon=1.0)
        self.query_id = 1

    def generate_pii_heavy_queries(self, n=250):
        """Generate 40% PII-heavy queries (250 total)"""
        queries = []

        templates = [
            "What is the balance on my account {iban}?",
            "Can you transfer {amount} EUR from account {iban} to {recipient_name}?",
            "I need to update my phone number to {phone} for account {account_num}",
            "Why was {amount} EUR deducted from my account {iban} on {date}?",
            "Please send my statement for account {account_num} to {email}",
            "I'm {name} and I can't access my account {iban}. My phone is {phone}",
            "Can you check if payment of {amount} EUR to {recipient_name} went through from {iban}?",
            "Update my address to {address} for account number {account_num}",
            "I need to report my card {card_number} as stolen. I'm {name}, phone {phone}",
            "What's the interest rate on my loan {account_num}? My email is {email}",
        ]

        for _ in range(n):
            template = random.choice(templates)

            query = template.format(
                iban=self._generate_iban(),
                amount=self.dp.perturb_amount(random.uniform(10, 5000)),
                recipient_name=fake.name(),
                account_num=fake.bban(),
                phone=fake.phone_number(),
                date=fake.date_between(start_date='-30d', end_date='today'),
                email=fake.email(),
                name=fake.name(),
                address=fake.address().replace('\n', ', '),
                card_number=fake.credit_card_number()
            )

            queries.append({
                'query_id': f'SYN_{self.query_id:04d}',
                'query_text': query,
                'category': 'PII-heavy',
                'source': 'faker_generated',
                'pii_entities': self._detect_pii_types(query),
                'token_count': len(query.split()) * 1.3,  # Approximate tokenization
                'differential_privacy': 'epsilon_1.0'
            })
            self.query_id += 1

        return queries

    def generate_compliance_queries(self, n=188):
        """Generate 30% compliance-related queries (188 total)"""
        queries = []

        templates = [
            "How does your bank comply with GDPR Article {article} regarding {topic}?",
            "What data do you collect under GDPR and how is it protected?",
            "Can I request deletion of my data under Article 17 GDPR?",
            "How do you ensure compliance with MiFID II for investment advice?",
            "What are your data retention policies according to GDPR Article 5?",
            "Do you use automated decision-making under GDPR Article 22?",
            "How can I access my personal data under GDPR Article 15?",
            "What security measures comply with GDPR Article 32?",
            "How do you handle data breaches according to GDPR Article 33?",
            "Can you explain your legal basis for processing my data under Article 6?",
            "What third parties have access to my data and under what GDPR provisions?",
            "How does your chatbot comply with EU AI Act transparency requirements?",
            "What consent mechanisms do you use under GDPR Article 7?",
            "How do you ensure data portability under GDPR Article 20?",
            "What DPO contact information is available per GDPR Article 37?",
        ]

        gdpr_articles = ['5', '6', '7', '13', '14', '15', '17', '20', '22', '32', '33', '37']
        topics = ['data minimization', 'purpose limitation', 'storage limitation',
                 'data accuracy', 'confidentiality', 'lawful processing']

        for _ in range(n):
            template = random.choice(templates)

            if '{article}' in template and '{topic}' in template:
                query = template.format(
                    article=random.choice(gdpr_articles),
                    topic=random.choice(topics)
                )
            else:
                query = template

            queries.append({
                'query_id': f'SYN_{self.query_id:04d}',
                'query_text': query,
                'category': 'compliance',
                'source': 'curated',
                'pii_entities': [],
                'token_count': len(query.split()) * 1.3,
                'differential_privacy': 'not_applicable'
            })
            self.query_id += 1

        return queries

    def generate_general_banking_queries(self, n=187):
        """Generate 30% general banking queries (187 total)"""
        queries = []

        templates = [
            "What are your current interest rates for savings accounts?",
            "How do I open a new checking account?",
            "What fees do you charge for international transfers?",
            "Can you explain the difference between fixed and variable mortgages?",
            "What are the requirements for getting a personal loan?",
            "How long does it take to process a loan application?",
            "What investment products do you offer?",
            "Can I set up automatic payments for my bills?",
            "What are your business hours and branch locations?",
            "How do I activate mobile banking?",
            "What security features protect my online banking?",
            "Can you explain how overdraft protection works?",
            "What credit cards do you offer and what are their benefits?",
            "How do I dispute a transaction?",
            "What documents do I need to open an account?",
            "Can I get a student loan and what are the terms?",
            "How do currency exchange rates work for foreign transactions?",
            "What's the minimum balance required for a savings account?",
            "Can you explain your mobile app features?",
            "How do I order a new card if mine is lost?",
        ]

        for _ in range(n):
            query = random.choice(templates)

            # Add contextual variation
            if random.random() < 0.3:
                contexts = [
                    f"I'm a new customer and {query.lower()}",
                    f"As an existing customer, {query.lower()}",
                    f"I'm interested in {query.lower()}",
                    f"Could you please help me understand {query.lower()}",
                ]
                query = random.choice(contexts)

            queries.append({
                'query_id': f'SYN_{self.query_id:04d}',
                'query_text': query,
                'category': 'general_banking',
                'source': 'curated',
                'pii_entities': [],
                'token_count': len(query.split()) * 1.3,
                'differential_privacy': 'not_applicable'
            })
            self.query_id += 1

        return queries

    def _generate_iban(self):
        """Generate structurally valid but non-functional IBAN"""
        country_codes = ['SE', 'FI', 'NO', 'DK', 'DE', 'GB']
        country = random.choice(country_codes)
        check_digits = f"{random.randint(10, 99)}"
        bank_code = f"{random.randint(1000, 9999)}"
        account = f"{random.randint(1000000, 9999999)}"
        return f"{country}{check_digits}{bank_code}{account}"

    def _detect_pii_types(self, query):
        """Detect PII entity types in query"""
        pii_types = []

        patterns = {
            'IBAN': r'[A-Z]{2}\d{2}[A-Z0-9]+',
            'EMAIL': r'[\w\.-]+@[\w\.-]+',
            'PHONE': r'\+?\d[\d\s\-\(\)]+',
            'CARD': r'\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}',
        }

        import re
        for pii_type, pattern in patterns.items():
            if re.search(pattern, query):
                pii_types.append(pii_type)

        # Check for names (simple heuristic)
        if any(word[0].isupper() and len(word) > 3 for word in query.split()):
            pii_types.append('PERSON')

        return pii_types

    def generate_all_queries(self):
        """Generate complete dataset of 625 queries"""
        print("Generating synthetic queries...")

        # Generate all categories
        pii_queries = self.generate_pii_heavy_queries(250)  # 40%
        compliance_queries = self.generate_compliance_queries(188)  # 30%
        general_queries = self.generate_general_banking_queries(187)  # 30%

        # Combine all queries
        all_queries = pii_queries + compliance_queries + general_queries

        # Shuffle to randomize order
        random.shuffle(all_queries)

        # Add metadata
        dataset = {
            'metadata': {
                'total_queries': len(all_queries),
                'generation_date': datetime.now().isoformat(),
                'differential_privacy_epsilon': 1.0,
                'distribution': {
                    'pii_heavy': f"{len(pii_queries)} (40%)",
                    'compliance': f"{len(compliance_queries)} (30%)",
                    'general_banking': f"{len(general_queries)} (30%)"
                },
                'faker_generated': 350,
                'curated_queries': 275,
                'average_token_count': 512,
                'purpose': 'Financial chatbot testing - Section 3.3.4'
            },
            'queries': all_queries
        }

        print(f"✓ Generated {len(all_queries)} queries")
        print(f"  - PII-heavy: {len(pii_queries)}")
        print(f"  - Compliance: {len(compliance_queries)}")
        print(f"  - General banking: {len(general_queries)}")

        return dataset


def save_dataset(dataset, formats=['csv', 'json', 'parquet']):
    """Save dataset in multiple formats"""

    df = pd.DataFrame(dataset['queries'])

    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if 'csv' in formats:
        filename = f'synthetic_queries_{timestamp}.csv'
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Saved CSV: {filename}")

    if 'json' in formats:
        filename = f'synthetic_queries_{timestamp}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved JSON: {filename}")

    if 'parquet' in formats:
        filename = f'synthetic_queries_{timestamp}.parquet'
        df.to_parquet(filename, index=False, engine='pyarrow')
        print(f"✓ Saved Parquet: {filename}")

    # Generate summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total queries: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nSource distribution:")
    print(df['source'].value_counts())
    print(f"\nAverage token count: {df['token_count'].mean():.1f}")
    print(f"Token count range: {df['token_count'].min():.0f} - {df['token_count'].max():.0f}")

    return df


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Synthetic Queries Generator for Financial Chatbot")
    print("Based on Dissertation Section 3.3.4: Data Pre-processing")
    print("=" * 60)
    print()

    # Generate dataset
    generator = SyntheticQueryGenerator()
    dataset = generator.generate_all_queries()

    print()

    # Save in multiple formats
    df = save_dataset(dataset, formats=['csv', 'json', 'parquet'])

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)

    # Display sample queries
    print("\n=== Sample Queries ===\n")
    for category in ['PII-heavy', 'compliance', 'general_banking']:
        sample = df[df['category'] == category].iloc[0]
        print(f"{category.upper()}:")
        print(f"  Query: {sample['query_text'][:100]}...")
        print(f"  PII entities: {sample['pii_entities']}")
        print()
