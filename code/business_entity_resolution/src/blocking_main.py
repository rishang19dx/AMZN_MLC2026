"""
Main Candidate Generation / Blocking CLI entry point.

Usage:
    python -m business_entity_resolution.src.blocking \
        --test-dir dataset/test \
        --artifacts-dir artifacts \
        --output-dir output \
        --config configs/blocking.yaml
"""

import argparse
import os
import yaml
import pandas as pd

from business_entity_resolution.src.blocking.bert_blocker import BERTBlocker
from business_entity_resolution.src.blocking.candidate_union import CandidateUnion
from business_entity_resolution.src.blocking.classical_blocker import ClassicalBlocker
from business_entity_resolution.src.blocking.learned_blocker import LearnedBlocker
from business_entity_resolution.src.data.loader import load_source_file
from business_entity_resolution.src.models.bert_encoder import BERTDualEncoder, HAS_TORCH_TRANSFORMERS
from business_entity_resolution.src.models.learned_encoder import JEPAEntityEncoder


def main():
    parser = argparse.ArgumentParser(description="Multi-Pipeline Candidate Generation and Blocking")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Path to artifacts directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to save outputs")
    parser.add_argument("--config", type=str, default="configs/blocking.yaml", help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/auto)")
    parser.add_argument("--max-candidates-per-entity", type=int, default=None, help="Candidate budget per S1 entity")
    parser.add_argument("--nrows", type=int, default=None, help="Optional limit on rows for quick testing")

    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        alt_cfg = os.path.join("business_entity_resolution", args.config)
        if os.path.exists(alt_cfg):
            with open(alt_cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

    blocking_cfg = cfg.get("blocking", {})
    models_cfg = cfg.get("models", {})
    device = args.device if args.device is not None else blocking_cfg.get("device", "auto")
    max_cands = (
        args.max_candidates_per_entity
        if args.max_candidates_per_entity is not None
        else blocking_cfg.get("max_candidates_per_entity", 100)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("Multi-Pipeline Candidate Generation / Blocking System")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Artifacts directory: {args.artifacts_dir}")
    print(f"Max candidates per S1 entity: {max_cands}")
    print("=" * 65)

    # 1. Load Data
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    print("\n[Step 1/5] Loading and normalizing test datasets...")
    s1_df = load_source_file(s1_path, nrows=args.nrows)
    s2_df = load_source_file(s2_path, nrows=args.nrows)
    s3_df = load_source_file(s3_path, nrows=args.nrows)
    print(f"Loaded records -> Source 1: {len(s1_df):,}, Source 2: {len(s2_df):,}, Source 3: {len(s3_df):,}")

    all_s1_ids = s1_df["entity_id"].tolist()

    bert_cands = {}
    learned_cands = {}
    classical_cands = {}

    # 2. Pipeline A: Fine-tuned BERT Representation
    bert_cfg = models_cfg.get("bert", {})
    if bert_cfg.get("enabled", True) and HAS_TORCH_TRANSFORMERS:
        print("\n[Step 2/5] Running Pipeline A: BERT Semantic Dual Encoder...")
        bert_model_dir = os.path.join(args.artifacts_dir, "models", "bert_encoder")
        model_to_load = (
            bert_model_dir
            if os.path.exists(bert_model_dir)
            else bert_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        print(f"Loading BERT encoder from: {model_to_load}")
        try:
            bert_encoder = BERTDualEncoder(
                model_name=model_to_load,
                max_length=bert_cfg.get("max_length", 128),
            )
            bert_blocker = BERTBlocker(
                encoder=bert_encoder,
                top_k_source2=bert_cfg.get("top_k_source2", 30),
                top_k_source3=bert_cfg.get("top_k_source3", 30),
                batch_size=bert_cfg.get("batch_size", 64),
                device=device,
                use_faiss=blocking_cfg.get("use_faiss", True),
            )
            print("Fitting ANN indices for S2 and S3...")
            bert_blocker.fit_targets(s2_df, s3_df, show_progress=True)
            print("Retrieving BERT candidates for S1...")
            bert_cands = bert_blocker.retrieve_candidates(s1_df, show_progress=True)
            print(f"BERT retrieved candidates for {len(bert_cands)} S1 entities.")
        except Exception as e:
            print(f"Warning: BERT blocker encountered error: {e}. Proceeding with remaining pipelines.")

    # 3. Pipeline B: Second Learned Representation / JEPA Blocker
    learned_cfg = models_cfg.get("learned", {})
    if learned_cfg.get("enabled", True) and HAS_TORCH_TRANSFORMERS:
        print("\n[Step 3/5] Running Pipeline B: Learned / JEPA Representation Blocker...")
        learned_model_dir = os.path.join(args.artifacts_dir, "models", "learned_encoder")
        model_to_load = (
            learned_model_dir
            if os.path.exists(learned_model_dir)
            else learned_cfg.get("model_name", "google/electra-small-discriminator")
        )
        print(f"Loading Learned encoder from: {model_to_load}")
        try:
            learned_encoder = JEPAEntityEncoder(
                model_name=model_to_load,
                max_length=learned_cfg.get("max_length", 128),
            )
            learned_blocker = LearnedBlocker(
                encoder=learned_encoder,
                top_k_source2=learned_cfg.get("top_k_source2", 30),
                top_k_source3=learned_cfg.get("top_k_source3", 30),
                batch_size=learned_cfg.get("batch_size", 64),
                device=device,
                use_faiss=blocking_cfg.get("use_faiss", True),
            )
            print("Fitting independent ANN indices for S2 and S3...")
            learned_blocker.fit_targets(s2_df, s3_df, show_progress=True)
            print("Retrieving Learned candidates for S1...")
            learned_cands = learned_blocker.retrieve_candidates(s1_df, show_progress=True)
            print(f"Learned blocker retrieved candidates for {len(learned_cands)} S1 entities.")
        except Exception as e:
            print(f"Warning: Learned blocker encountered error: {e}. Proceeding with remaining pipelines.")

    # 4. Pipeline C: Classical Non-Neural Blocker
    classical_cfg = models_cfg.get("classical", {})
    if classical_cfg.get("enabled", True):
        print("\n[Step 4/5] Running Pipeline C: Classical Multi-Key Inverted Index Blocker...")
        classical_blocker = ClassicalBlocker(
            top_k_source2=classical_cfg.get("top_k_source2", 30),
            top_k_source3=classical_cfg.get("top_k_source3", 30),
            max_bucket_size=classical_cfg.get("max_bucket_size", 2000),
            min_score_threshold=classical_cfg.get("min_score_threshold", 0.15),
        )
        print("Fitting inverted multi-key index on S2 and S3...")
        classical_blocker.fit_targets(s2_df, s3_df)
        print("Retrieving classical candidates for S1...")
        classical_cands = classical_blocker.retrieve_candidates(s1_df)
        print(f"Classical blocker retrieved candidates for {len(classical_cands)} S1 entities.")

    # 5. Candidate Union & Deduplication
    print("\n[Step 5/5] Performing Candidate Union (C_bert UNION C_learned UNION C_classical)...")
    union = CandidateUnion(max_candidates_per_entity=max_cands)
    final_candidates, debug_df = union.merge_candidates(
        all_s1_ids=all_s1_ids,
        bert_candidates=bert_cands,
        learned_candidates=learned_cands,
        classical_candidates=classical_cands,
    )

    # Export candidate_pairs.tsv
    cand_out_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    print(f"Exporting final candidate pairs to: {cand_out_path}")
    union.export_candidate_pairs(final_candidates, cand_out_path)

    # Export debug provenance
    debug_out_path = os.path.join(args.output_dir, "debug_candidate_scores.tsv")
    print(f"Exporting debug provenance to: {debug_out_path}")
    union.export_debug_scores(debug_df, debug_out_path)

    # Optional LightGBM submission filter
    filter_cfg = cfg.get("filtering", {})
    if filter_cfg.get("enabled", False):
        print("\n--- Optional LightGBM Submission Filter ---")
        try:
            from business_entity_resolution.src.lightgbm_filter import LightGBMFilter
            lgb_filter = LightGBMFilter(threshold=float(filter_cfg.get("threshold", 0.5)))
            # If pretrained model exists in artifacts, load it
            target_combined = pd.concat([s2_df, s3_df], ignore_index=True)
            matches = lgb_filter.filter_candidates(
                final_candidates, s1_df, target_combined, provenance_df=debug_df
            )
            matching_out_path = os.path.join(args.output_dir, "matching_results.tsv")
            lgb_filter.export_matching_results(matches, matching_out_path)
            print(f"Exported matching results to: {matching_out_path}")
        except Exception as e:
            print(f"Note: LightGBM filter skipped: {e}")

    print("\n" + "=" * 65)
    print("Multi-Pipeline Candidate Generation Completed Successfully!")
    print(f"Candidate output: {cand_out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
