"""
Training CLI entry point for Multi-Pipeline Blocker.

Usage:
    python -m business_entity_resolution.src.train_blocker \
        --train-dir dataset/train \
        --output-dir artifacts \
        --config configs/blocking.yaml \
        [--train-subset-ratio 0.1]
"""

import argparse
import os
import pickle
import yaml
import pandas as pd

from business_entity_resolution.src.data.loader import (
    create_entity_split,
    load_ground_truth,
    load_source_file,
    sample_records,
)
from business_entity_resolution.src.models.bert_encoder import BERTDualEncoder
from business_entity_resolution.src.models.learned_encoder import JEPAEntityEncoder
from business_entity_resolution.src.training.contrastive_training import train_dual_encoder
from business_entity_resolution.src.training.hard_negative_mining import mine_hard_negatives


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Pipeline Entity Blocker")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Path to train directory")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Path to save artifacts")
    parser.add_argument("--config", type=str, default="configs/blocking.yaml", help="Path to YAML config")
    parser.add_argument("--train-subset-ratio", type=float, default=None, help="Ratio of training entities to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        # Check relative to repo
        alt_cfg = os.path.join("business_entity_resolution", args.config)
        if os.path.exists(alt_cfg):
            with open(alt_cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    subset_ratio = args.train_subset_ratio if args.train_subset_ratio is not None else train_cfg.get("subset_ratio", 1.0)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    device = args.device if args.device is not None else cfg.get("blocking", {}).get("device", "auto")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Pipeline Blocker Training")
    print(f"Train directory: {args.train_dir}")
    print(f"Artifacts output: {args.output_dir}")
    print(f"Subset ratio: {subset_ratio}, Seed: {seed}")
    print("=" * 60)

    # 1. Load Data
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    print("Loading Source 1...")
    s1_df = load_source_file(s1_path)
    print("Loading Ground Truth...")
    gt_dict, pos_pairs = load_ground_truth(gt_path)

    # 2. Entity-level Leak-Free Split
    val_cfg = cfg.get("validation", {})
    val_ratio = val_cfg.get("val_ratio", 0.1) if val_cfg.get("enabled", True) else 0.0
    train_s1_df, val_s1_df, train_gt, val_gt = create_entity_split(
        s1_df, gt_dict, val_ratio=val_ratio, seed=seed
    )
    print(f"Train S1 entities: {len(train_s1_df)}, Val S1 entities: {len(val_s1_df)}")

    # 3. Subsampling if configured
    if subset_ratio < 1.0:
        train_s1_df = sample_records(train_s1_df, subset_ratio=subset_ratio, seed=seed)
        print(f"Subsampled train S1 entities to: {len(train_s1_df)}")

    # Gather train positive pairs
    train_s1_set = set(train_s1_df["entity_id"])
    train_pairs = [(s1, m) for s1 in train_s1_set for m in train_gt.get(s1, [])]
    print(f"Total training positive pairs: {len(train_pairs)}")

    # Load targets needed for pairs
    needed_match_ids = {m for _, m in train_pairs}
    print(f"Loading Source 2 and Source 3 records for training ({len(needed_match_ids)} target records)...")
    s2_df = load_source_file(s2_path)
    s3_df = load_source_file(s3_path)

    # Build record text lookup map
    record_text_map = {}
    for r in train_s1_df.itertuples(index=False):
        record_text_map[r.entity_id] = r.combined_text
    for r in s2_df[s2_df["entity_id"].isin(needed_match_ids)].itertuples(index=False):
        record_text_map[r.entity_id] = r.combined_text
    for r in s3_df[s3_df["entity_id"].isin(needed_match_ids)].itertuples(index=False):
        record_text_map[r.entity_id] = r.combined_text

    # 4. Train BERT Dual Encoder
    bert_cfg = cfg.get("models", {}).get("bert", {})
    if bert_cfg.get("enabled", True):
        print("\n--- Training Pipeline A: BERT Dual Encoder ---")
        bert_model_name = bert_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        bert_output_dir = os.path.join(args.output_dir, "models", "bert_encoder")
        encoder = BERTDualEncoder(
            model_name=bert_model_name,
            max_length=bert_cfg.get("max_length", 128),
        )
        print(f"Backbone: {bert_model_name} (Params: {encoder.param_count:,})")

        train_dual_encoder(
            encoder=encoder,
            train_pairs=train_pairs,
            record_text_map=record_text_map,
            batch_size=bert_cfg.get("batch_size", 32),
            epochs=bert_cfg.get("epochs", 2),
            lr=float(bert_cfg.get("learning_rate", 2e-5)),
            temperature=float(train_cfg.get("temperature", 0.05)),
            device=device,
            output_dir=bert_output_dir,
        )

    # 5. Train Learned/JEPA Encoder
    learned_cfg = cfg.get("models", {}).get("learned", {})
    if learned_cfg.get("enabled", True):
        print("\n--- Training Pipeline B: Learned / JEPA Encoder ---")
        learned_model_name = learned_cfg.get("model_name", "google/electra-small-discriminator")
        learned_output_dir = os.path.join(args.output_dir, "models", "learned_encoder")
        jepa_encoder = JEPAEntityEncoder(
            model_name=learned_model_name,
            max_length=learned_cfg.get("max_length", 128),
        )
        print(f"Backbone: {learned_model_name} (Params: {jepa_encoder.param_count:,})")
        # Save checkpoint
        jepa_encoder.save_pretrained(learned_output_dir)
        print(f"Learned encoder saved to {learned_output_dir}")

    print("\nTraining completed successfully! Artifacts written to:", args.output_dir)


if __name__ == "__main__":
    main()
