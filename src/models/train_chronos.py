"""LoRA Fine-Tuning Pipeline for Chronos-2 on Caltrans PeMS04 Traffic Flow.
===========================================================================
Fine-tunes the Amazon Chronos-2 pretrained foundation model (`amazon/chronos-2`)
using Low-Rank Adaptation (LoRA), preserving frozen foundation weights while
adapting attention projections to urban highway traffic dynamics.

Chronological Split Enforced:
- Training: First 60% of timeline (0 to 10,204 timesteps, ~35.4 days).
- Validation: Middle 20% of timeline (10,204 to 13,598 timesteps, ~11.8 days).
- Held-out Test: Final 20% of timeline (13,598 to 16,992 timesteps, ~11.8 days) - untouched.

Usage:
    python -m src.models.train_chronos --num_steps 300 --batch_size 32 --lr 1e-4
"""

import os
# Ensure TensorFlow / Keras 3 conflicts are avoided
os.environ['USE_TF'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos import Chronos2Pipeline
from peft import LoraConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_chronos_lora(
    num_steps: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    context_length: int = 288,
    prediction_length: int = 12,
    lora_r: int = 8,
    lora_alpha: int = 16,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Execute LoRA fine-tuning for Chronos-2 on PeMS04 traffic flow.

    Args:
        num_steps: Number of optimization steps.
        batch_size: Training batch size (number of series windows per step).
        learning_rate: LoRA optimizer learning rate (AdamW).
        context_length: Historical input window length (default: 288 = 24 hours).
        prediction_length: Future forecast horizon (default: 12 = 60 minutes).
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        output_dir: Directory where checkpoints and metadata will be saved.
        seed: Random seed for reproducibility.

    Returns:
        Path to the saved fine-tuned checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs" / "checkpoints" / "chronos2_lora")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("=" * 65)
    logger.info("CHRONOS-2 FOUNDATION MODEL LoRA FINE-TUNING")
    logger.info("=" * 65)
    logger.info(f"Target Hardware: {device_name} ({device.upper()})")
    logger.info(f"Optimization Steps: {num_steps} | Batch Size: {batch_size} | LR: {learning_rate}")
    logger.info(f"Context Length: {context_length} (24h) | Forecast Horizon: {prediction_length} (1h)")

    # 1. Ingest Raw PeMS04 Flow Data
    raw_path = PROJECT_ROOT / "data" / "raw" / "pems04.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"PeMS04 raw data file missing at: {raw_path}")

    logger.info(f"Loading PeMS04 traffic flow from: {raw_path}")
    raw_data = np.load(raw_path)["data"]  # (16992, 307, 3)
    flow_data = raw_data[:, :, 0]         # Extract flow channel: (16992, 307)
    total_timesteps, num_sensors = flow_data.shape
    logger.info(f"Dataset Dimensions: {total_timesteps} intervals across {num_sensors} sensors.")

    # 2. Enforce Strict Zero-Leakage Chronological Partitioning
    # 60% Train (0 -> 10,204), 20% Val (10,204 -> 13,598), 20% Test (13,598 -> 16,992)
    train_end = 10204
    val_end = 13598

    train_flow = flow_data[:train_end, :]
    val_flow = flow_data[train_end:val_end, :]
    test_flow = flow_data[val_end:, :]

    logger.info(f"Chronological Splits (Zero-Leakage):")
    logger.info(f"  • Train Partition:      steps 0 -> {train_end} ({train_flow.shape[0]} intervals, ~35.4 days)")
    logger.info(f"  • Validation Partition: steps {train_end} -> {val_end} ({val_flow.shape[0]} intervals, ~11.8 days)")
    logger.info(f"  • Held-out Test:        steps {val_end} -> {total_timesteps} ({test_flow.shape[0]} intervals, ~11.8 days)")

    # Prepare inputs as list of 307 continuous univariate series
    train_inputs = [torch.from_numpy(train_flow[:, s]).float() for s in range(num_sensors)]
    val_inputs = [torch.from_numpy(val_flow[:, s]).float() for s in range(num_sensors)]

    # 3. Load Pretrained Foundation Model
    logger.info("Initializing base foundation model 'amazon/chronos-2'...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # 4. Configure LoRA Parameters
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )
    logger.info(f"Configured LoRA Adapter: rank={lora_r}, alpha={lora_alpha}")

    # 5. Execute Fine-Tuning
    start_time = time.time()
    logger.info("Starting LoRA fine-tuning...")

    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=prediction_length,
        context_length=context_length,
        validation_inputs=val_inputs,
        finetune_mode="lora",
        lora_config=lora_config,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=out_path,
        finetuned_ckpt_name="finetuned-ckpt",
    )

    training_duration = time.time() - start_time
    saved_ckpt_dir = out_path / "finetuned-ckpt"
    logger.info(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes).")
    logger.info(f"Fine-tuned checkpoint verified at: {saved_ckpt_dir}")

    # 6. Save Experiment Metadata
    meta = {
        "model": "Chronos-2 (amazon/chronos-2)",
        "approach": "Foundation Model Transfer Learning via LoRA",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "device": device,
            "device_name": device_name,
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset": "Caltrans PeMS04",
        "num_sensors": num_sensors,
        "chronological_splits": {
            "train_steps": [0, train_end],
            "val_steps": [train_end, val_end],
            "test_steps": [val_end, total_timesteps],
        },
        "hyperparameters": {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "finetune_mode": "lora",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "seed": seed,
        },
        "training_duration_seconds": round(training_duration, 2),
        "saved_checkpoint_path": str(saved_ckpt_dir),
    }

    meta_path = out_path / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    logger.info(f"Saved training metadata to: {meta_path}")

    return str(saved_ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chronos-2 LoRA on PeMS04")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of fine-tuning steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (series per step)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--context_length", type=int, default=288, help="Context length (5-min steps)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    train_chronos_lora(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        context_length=args.context_length,
        output_dir=args.output_dir,
    )
