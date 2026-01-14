#!/usr/bin/env python
"""Test script for Vajra BM25 integration."""

import sys
sys.path.insert(0, '/Users/rajesh/Github/praval_deep_research/src')

from datetime import datetime, timezone

def main():
    from agentic_research.storage.conversation_index import get_conversation_index

    # Get the conversation index
    index = get_conversation_index()

    # Add some test messages
    print("Indexing test messages...")
    now = datetime.now(timezone.utc)

    index.index_message(
        message_id="test-msg-001",
        conversation_id="test-conv-001",
        user_id="test-user",
        role="user",
        content="What is machine learning and how does it work?",
        timestamp=now,
    )

    index.index_message(
        message_id="test-msg-002",
        conversation_id="test-conv-001",
        user_id="test-user",
        role="assistant",
        content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        timestamp=now,
    )

    index.index_message(
        message_id="test-msg-003",
        conversation_id="test-conv-002",
        user_id="test-user",
        role="user",
        content="Explain neural networks and deep learning architectures.",
        timestamp=now,
    )

    print(f"Indexed {index.document_count} messages")

    # Build the index
    print("\nBuilding BM25 index...")
    index.rebuild_index()

    # Search for messages
    print("\n--- Search Results for 'machine learning' ---")
    results = index.search_conversations(query="machine learning", top_k=5)
    for hit in results:
        print(f"  [{hit.metadata.get('role')}] Score: {hit.score:.3f}")
        print(f"    {hit.content[:80]}...")

    print("\n--- Search Results for 'neural networks' ---")
    results = index.search_conversations(query="neural networks", top_k=5)
    for hit in results:
        print(f"  [{hit.metadata.get('role')}] Score: {hit.score:.3f}")
        print(f"    {hit.content[:80]}...")

    # Test the Praval tool
    print("\n--- Testing search_conversations Tool ---")
    from tools.search_conversations import search_conversations

    tool_results = search_conversations(query="artificial intelligence", top_k=3)
    print(f"Tool returned {len(tool_results)} results:")
    for r in tool_results:
        print(f"  [{r['role']}] Score: {r['score']}")
        print(f"    {r['content'][:60]}...")

    print("\n✅ Vajra BM25 integration working!")

if __name__ == "__main__":
    main()
