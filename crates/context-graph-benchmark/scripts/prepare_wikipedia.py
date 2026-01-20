#!/usr/bin/env python3
"""
Prepare Wikipedia dataset for benchmark testing.

Downloads English Wikipedia from HuggingFace, chunks the text,
and saves in a format ready for the Rust benchmark loader.

Usage:
    python prepare_wikipedia.py --output /path/to/output --max-docs 100000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, List, Tuple
import hashlib

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Tuple[str, int, int]]:
    """
    Chunk text into overlapping segments of approximately chunk_size words.

    Returns list of (chunk_text, start_word_idx, end_word_idx) tuples.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [(text, 0, len(words))]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append((chunk_text, start, end))

        # Move forward by (chunk_size - overlap) words
        start += chunk_size - overlap

        # Don't create tiny final chunks
        if len(words) - start < overlap:
            break

    return chunks


def generate_chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Generate a deterministic UUID-like ID for a chunk."""
    content = f"{doc_id}:{chunk_idx}"
    hash_bytes = hashlib.sha256(content.encode()).digest()[:16]
    # Format as UUID
    return f"{hash_bytes[:4].hex()}-{hash_bytes[4:6].hex()}-{hash_bytes[6:8].hex()}-{hash_bytes[8:10].hex()}-{hash_bytes[10:16].hex()}"


def process_wikipedia(
    output_dir: Path,
    max_docs: int = 100000,
    chunk_size: int = 200,
    overlap: int = 50,
    min_article_words: int = 100,
) -> dict:
    """
    Download and process Wikipedia dataset.

    Returns statistics about the processed data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = output_dir / "chunks.jsonl"
    metadata_file = output_dir / "metadata.json"

    print(f"Loading Wikipedia dataset (this may take a while on first run)...")
    print(f"  Max documents: {max_docs}")
    print(f"  Chunk size: {chunk_size} words")
    print(f"  Overlap: {overlap} words")
    print(f"  Min article words: {min_article_words}")
    print()

    # Load Wikipedia dataset
    # Using streaming to avoid downloading everything at once
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    stats = {
        "total_documents": 0,
        "total_chunks": 0,
        "total_words": 0,
        "skipped_short": 0,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "source": "wikimedia/wikipedia:20231101.en",
    }

    # Topic tracking for ground truth
    # We'll use Wikipedia categories as proxy for topics
    topic_counts = {}

    print("Processing articles...")
    with open(chunks_file, 'w') as f:
        for i, article in enumerate(dataset):
            if i >= max_docs:
                break

            if i > 0 and i % 10000 == 0:
                print(f"  Processed {i:,} articles, {stats['total_chunks']:,} chunks...")

            text = article.get('text', '')
            title = article.get('title', f'article_{i}')
            doc_id = article.get('id', str(i))

            # Skip short articles
            words = text.split()
            if len(words) < min_article_words:
                stats["skipped_short"] += 1
                continue

            # Extract simple topic from title (first word or category hint)
            # This is a rough approximation - real topic detection would use categories
            topic = title.split()[0].lower() if title else "unknown"
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # Chunk the article
            chunks = chunk_text(text, chunk_size, overlap)

            for chunk_idx, (chunk_text, start_word, end_word) in enumerate(chunks):
                chunk_id = generate_chunk_id(doc_id, chunk_idx)

                chunk_record = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "start_word": start_word,
                    "end_word": end_word,
                    "topic_hint": topic,
                }

                f.write(json.dumps(chunk_record) + '\n')
                stats["total_chunks"] += 1
                stats["total_words"] += len(chunk_text.split())

            stats["total_documents"] += 1

    # Find top topics for ground truth clustering
    sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:100]
    stats["top_topics"] = [t[0] for t in sorted_topics]
    stats["topic_counts"] = dict(sorted_topics)

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print()
    print("Processing complete!")
    print(f"  Documents processed: {stats['total_documents']:,}")
    print(f"  Documents skipped (too short): {stats['skipped_short']:,}")
    print(f"  Chunks generated: {stats['total_chunks']:,}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Avg words per chunk: {stats['total_words'] / max(1, stats['total_chunks']):.1f}")
    print()
    print(f"Output files:")
    print(f"  Chunks: {chunks_file}")
    print(f"  Metadata: {metadata_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia dataset for benchmark testing"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./benchmark_data/wikipedia",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-docs", "-n",
        type=int,
        default=100000,
        help="Maximum number of documents to process (default: 100000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Target chunk size in words (default: 200)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in words (default: 50)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=100,
        help="Minimum article length in words (default: 100)"
    )

    args = parser.parse_args()

    process_wikipedia(
        output_dir=args.output,
        max_docs=args.max_docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_article_words=args.min_words,
    )


if __name__ == "__main__":
    main()
