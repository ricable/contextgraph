#!/usr/bin/env python3
"""Convert pytorch_model.bin files to model.safetensors format.

This script converts the following models that only have pytorch_model.bin:
- sparse (naver/splade-cocondenser-ensembledistil)
- code (Salesforce/codet5p-110m-embedding)
- causal (allenai/longformer-base-4096)

Usage:
    python scripts/convert_pytorch_to_safetensors.py

Requirements:
    pip install torch safetensors
"""

import os
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install torch safetensors")
    sys.exit(1)


def break_shared_tensors(weights: dict) -> dict:
    """Break shared tensors by cloning them to avoid safetensors errors.

    Some HuggingFace models have tied weights (e.g., embedding/unembedding).
    SafeTensors doesn't support shared memory, so we need to clone them.

    Args:
        weights: Dictionary of tensor name -> tensor

    Returns:
        Dictionary with all shared tensors cloned to be independent
    """
    # First, identify all tensors and their data pointers
    data_ptr_to_tensors = {}
    for name, tensor in weights.items():
        if isinstance(tensor, torch.Tensor):
            ptr = tensor.data_ptr()
            if ptr not in data_ptr_to_tensors:
                data_ptr_to_tensors[ptr] = []
            data_ptr_to_tensors[ptr].append(name)

    # Find shared tensors
    shared_groups = [names for names in data_ptr_to_tensors.values() if len(names) > 1]

    if shared_groups:
        print(f"  Found {len(shared_groups)} groups of shared tensors, cloning...")
        for group in shared_groups:
            print(f"    Shared: {group}")

    # Clone tensors that share memory (keep first, clone rest)
    result = {}
    for name, tensor in weights.items():
        if isinstance(tensor, torch.Tensor):
            ptr = tensor.data_ptr()
            tensor_names = data_ptr_to_tensors.get(ptr, [])
            # If this tensor shares memory and it's not the first in the group, clone it
            if len(tensor_names) > 1 and tensor_names.index(name) > 0:
                result[name] = tensor.clone().contiguous()
            else:
                result[name] = tensor.contiguous()
        else:
            result[name] = tensor

    return result


def convert_model(model_name: str, model_dir: Path) -> bool:
    """Convert a single model from pytorch_model.bin to model.safetensors.

    Args:
        model_name: Name of the model (for logging)
        model_dir: Path to model directory

    Returns:
        True if conversion succeeded, False otherwise
    """
    pytorch_path = model_dir / "pytorch_model.bin"
    safetensors_path = model_dir / "model.safetensors"

    # Check if already converted
    if safetensors_path.exists():
        print(f"[{model_name}] model.safetensors already exists, skipping")
        return True

    # Check if source exists
    if not pytorch_path.exists():
        print(f"[{model_name}] ERROR: pytorch_model.bin not found at {pytorch_path}")
        return False

    print(f"[{model_name}] Converting pytorch_model.bin to model.safetensors...")
    print(f"  Source: {pytorch_path} ({pytorch_path.stat().st_size / 1024 / 1024:.1f} MB)")

    try:
        # Load PyTorch weights
        weights = torch.load(pytorch_path, map_location="cpu", weights_only=True)

        # Handle different weight formats
        if isinstance(weights, dict):
            if "state_dict" in weights:
                weights = weights["state_dict"]
            elif "model_state_dict" in weights:
                weights = weights["model_state_dict"]

        # Filter out non-tensor entries
        tensor_weights = {k: v for k, v in weights.items() if isinstance(v, torch.Tensor)}

        print(f"  Found {len(tensor_weights)} tensors")

        # Break shared tensors (clone them to be independent)
        tensor_weights = break_shared_tensors(tensor_weights)

        # Save as safetensors
        save_file(tensor_weights, safetensors_path)

        print(f"  Output: {safetensors_path} ({safetensors_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"[{model_name}] Conversion complete!")
        return True

    except Exception as e:
        print(f"[{model_name}] ERROR: Conversion failed: {e}")
        # Clean up partial file if it exists
        if safetensors_path.exists():
            safetensors_path.unlink()
        return False


def main():
    """Convert all models that need conversion."""
    # Get the models directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models"

    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        sys.exit(1)

    # Models that need conversion (only have pytorch_model.bin)
    models_to_convert = ["sparse", "code", "causal"]

    print("=" * 60)
    print("PyTorch to SafeTensors Conversion Script")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Models to convert: {models_to_convert}")
    print()

    results = {}
    for model_name in models_to_convert:
        model_dir = models_dir / model_name
        if not model_dir.exists():
            print(f"[{model_name}] WARNING: Model directory not found, skipping")
            results[model_name] = False
            continue

        results[model_name] = convert_model(model_name, model_dir)
        print()

    # Summary
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for model_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model_name}: {status}")

    print()
    print(f"Total: {success_count}/{total_count} succeeded")

    if success_count < total_count:
        sys.exit(1)

    # Verify files exist
    print()
    print("Verifying safetensors files...")
    for model_name in models_to_convert:
        safetensors_path = models_dir / model_name / "model.safetensors"
        if safetensors_path.exists():
            print(f"  {model_name}/model.safetensors: {safetensors_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {model_name}/model.safetensors: MISSING")


if __name__ == "__main__":
    main()
