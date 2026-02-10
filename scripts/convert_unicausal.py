#!/usr/bin/env python3
"""Convert UniCausal CSV splits to CausalTrainingPair JSONL format.

Usage:
    python scripts/convert_unicausal.py \
        --input data/causal/unicausal/data/splits/ \
        --output data/causal/unicausal_pairs.jsonl

Extracts <ARG0>cause</ARG0> and <ARG1>effect</ARG1> annotations from
the text_w_pairs column. Only includes pair_label=1 (confirmed causal).
"""

import csv
import json
import re
import sys
from pathlib import Path

ARG0_RE = re.compile(r'<ARG0>(.*?)</ARG0>')
ARG1_RE = re.compile(r'<ARG1>(.*?)</ARG1>')

# Map corpus names to domains
CORPUS_DOMAIN = {
    'ctb': 'news',
    'because': 'general',
    'altlex': 'news',
    'esl2': 'scientific',
    'semeval2010t8': 'general',
}


def extract_pair(text_w_pairs: str):
    """Extract cause (ARG0) and effect (ARG1) from annotated text."""
    causes = ARG0_RE.findall(text_w_pairs)
    effects = ARG1_RE.findall(text_w_pairs)
    if not causes or not effects:
        return None, None
    # Join multiple spans with space
    cause = ' '.join(c.strip() for c in causes)
    effect = ' '.join(e.strip() for e in effects)
    return cause, effect


def process_csv(csv_path: Path, pairs: list):
    """Process a single UniCausal CSV split file."""
    corpus = csv_path.stem.replace('_train', '').replace('_test', '')
    domain = CORPUS_DOMAIN.get(corpus, 'general')

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include confirmed causal pairs
            if row.get('pair_label', '0').strip() != '1':
                continue

            text_w_pairs = row.get('text_w_pairs', '')
            cause, effect = extract_pair(text_w_pairs)
            if not cause or not effect:
                continue

            # Skip very short spans (single word fragments)
            if len(cause) < 5 or len(effect) < 5:
                continue

            # Get full sentence for hard negative context
            full_text = row.get('text', '')

            pair = {
                'cause_text': cause,
                'effect_text': effect,
                'direction': 'Forward',
                'confidence': 0.85,  # External data: slightly lower confidence
                'mechanism': f'unicausal_{corpus}',
                'hard_negative': full_text if full_text != cause and full_text != effect else '',
                'domain': domain,
            }
            pairs.append(pair)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert UniCausal to JSONL')
    parser.add_argument('--input', default='data/causal/unicausal/data/splits/',
                        help='Directory containing UniCausal CSV splits')
    parser.add_argument('--output', default='data/causal/unicausal_pairs.jsonl',
                        help='Output JSONL file')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    pairs = []
    csv_files = sorted(input_dir.glob('*.csv'))
    for csv_path in csv_files:
        count_before = len(pairs)
        process_csv(csv_path, pairs)
        added = len(pairs) - count_before
        print(f"  {csv_path.name}: +{added} pairs")

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\nTotal: {len(pairs)} causal pairs -> {output_path}")


if __name__ == '__main__':
    main()
