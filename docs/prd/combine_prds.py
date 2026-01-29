"""Combine all 10 PRD files into a single markdown document."""

import os

prd_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(prd_dir, "PRD_COMBINED.md")

files = [f"PRD_{i:02d}_" for i in range(1, 11)]

with open(output_file, "w") as out:
    for i, prefix in enumerate(files):
        # Find the matching file
        matching = [f for f in sorted(os.listdir(prd_dir)) if f.startswith(prefix) and f.endswith(".md")]
        if not matching:
            print(f"Warning: No file found with prefix {prefix}")
            continue
        filepath = os.path.join(prd_dir, matching[0])
        with open(filepath, "r") as f:
            content = f.read()
        if i > 0:
            out.write("\n\n---\n\n")
        out.write(content)
    print(f"Combined PRD written to {output_file}")
