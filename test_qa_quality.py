"""
Demonstration of Q&A quality - showing the system works like Claude/ChatGPT research.
"""

import requests
import json

def test_qa_quality():
    """Test various questions to show answer quality."""

    questions = [
        "What are transformers in machine learning?",
        "How do attention mechanisms work?",
        "What are the advantages of transformers over RNNs?",
        "What are the main challenges with transformer architectures?",
        "How can transformers be applied to natural language processing?"
    ]

    print("=" * 80)
    print("Q&A Quality Demonstration")
    print("=" * 80)
    print()

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print('='*80)

        response = requests.post(
            "http://localhost:8000/research/ask",
            json={"question": question},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            sources = data['sources']
            confidence = data['confidence_score']

            print(f"\n📝 Answer (Confidence: {confidence:.2f}):")
            print("-" * 80)
            print(answer)
            print()

            print(f"📚 Sources Used: {len(sources)}")
            for j, source in enumerate(sources[:3], 1):
                print(f"  {j}. {source['title']} (relevance: {source['relevance_score']:.2f})")

            print()
            print(f"💡 Follow-up Questions:")
            for fq in data['followup_questions']:
                print(f"  - {fq}")

        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)

        print()

if __name__ == "__main__":
    test_qa_quality()
