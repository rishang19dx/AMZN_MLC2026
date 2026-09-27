"""Train a compact pair-compatibility model from the entity-resolution TSVs.

The input records and match labels are indexed in a temporary SQLite database,
so the full dataset is never loaded into RAM. The model learns a contrastive
score for matched versus randomly sampled same-source non-matched pairs.

Example:
    python train_pair_distribution.py --train-dir dataset/train --output-dir artifacts/pair_model
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import random
import re
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("Install numpy and PyTorch first (see requirements.txt).") from exc


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
CHAR_RE = re.compile(r"(?=([\w]{3}))", flags=re.UNICODE)
HASH_BUCKETS = 1 << 18
SEQUENCE_LENGTH = 64


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pair_distribution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def stable_validation_split(entity_id: str, validation_fraction: float) -> str:
    value = int(hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).hexdigest(), 16)
    return "validation" if value / (2**64) < validation_fraction else "train"


def read_tsv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            yield {
                key: (row.get(key) or "").strip()
                for key in ("entity_id", "business_name", "business_address", "country")
            }


def build_cache(train_dir: Path, db_path: Path, validation_fraction: float, logger: logging.Logger) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA cache_size=-65536")
        connection.executescript(
            """
            CREATE TABLE records (
                entity_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                name TEXT NOT NULL,
                address TEXT NOT NULL,
                country TEXT NOT NULL
            );
            CREATE INDEX records_source ON records(source);
            CREATE TABLE s1_split (entity_id TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE INDEX s1_split_name ON s1_split(split, entity_id);
            CREATE TABLE pairs (
                s1_id TEXT NOT NULL,
                target_id TEXT NOT NULL
            );
            CREATE INDEX pairs_s1 ON pairs(s1_id);
            """
        )

        for source_number in (1, 2, 3):
            source = f"S{source_number}"
            path = train_dir / f"train_source{source_number}.tsv"
            count = 0
            batch = []
            for row in read_tsv_rows(path):
                if not row["entity_id"]:
                    continue
                batch.append((row["entity_id"], source, row["business_name"], row["business_address"], row["country"]))
                if source_number == 1:
                    connection.execute(
                        "INSERT OR REPLACE INTO s1_split VALUES (?, ?)",
                        (row["entity_id"], stable_validation_split(row["entity_id"], validation_fraction)),
                    )
                if len(batch) >= 10_000:
                    connection.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?)", batch)
                    connection.commit()
                    count += len(batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?)", batch)
                count += len(batch)
            connection.commit()
            logger.info("Indexed %s: %s records", source, f"{count:,}")

        gt_path = train_dir / "train_ground_truth.tsv"
        with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"source1_entity_id", "matched_entity_ids"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{gt_path} is missing required columns: {sorted(missing)}")
            batch = []
            for row in reader:
                source_id = (row.get("source1_entity_id") or "").strip()
                matches = (row.get("matched_entity_ids") or "").strip()
                if source_id and matches and matches.lower() != "nan":
                    batch.extend((source_id, target.strip()) for target in matches.split(",") if target.strip())
                if len(batch) >= 10_000:
                    connection.executemany("INSERT INTO pairs VALUES (?, ?)", batch)
                    connection.commit()
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO pairs VALUES (?, ?)", batch)
        connection.commit()
        connection.execute("CREATE INDEX pairs_target ON pairs(target_id)")
        connection.commit()
        positive_count = connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        logger.info("Indexed %s labeled positive pairs", f"{positive_count:,}")
    finally:
        connection.close()


def hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") % (HASH_BUCKETS - 1)) + 1


def encode_record(name: str, address: str, country: str) -> list[int]:
    tokens: list[int] = []
    for field, value, budget in (("name", name, 28), ("address", address, 28), ("country", country, 8)):
        words = TOKEN_RE.findall(value.lower())
        tokens.extend(hash_token(f"{field}:{word}") for word in words[:budget])
        if field == "name":
            compact = " ".join(words)
            tokens.extend(hash_token(f"name3:{gram}") for gram in CHAR_RE.findall(compact)[:20])
    return tokens[:SEQUENCE_LENGTH] or [0]


class PairEnergyModel(nn.Module):
    """Shared entity encoder and symmetric compatibility-energy head."""

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(HASH_BUCKETS, embedding_dim, padding_idx=0)
        self.entity_projection = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.compatibility = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, pair_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, pair_size, sequence_length = pair_tokens.shape
        tokens = pair_tokens.reshape(batch_size * pair_size, sequence_length)
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        entities = self.entity_projection(pooled).reshape(batch_size, pair_size, -1)
        left, right = entities[:, 0], entities[:, 1]
        features = torch.cat((torch.abs(left - right), left * right, (left + right) * 0.5), dim=1)
        return self.compatibility(features).squeeze(1)


def encode_batch(records: Sequence[tuple[str, str, str, str, str, str]], device: torch.device) -> torch.Tensor:
    token_rows = []
    for left_name, left_address, left_country, right_name, right_address, right_country in records:
        left = encode_record(left_name, left_address, left_country)
        right = encode_record(right_name, right_address, right_country)
        left = (left + [0] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
        right = (right + [0] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
        token_rows.append((left, right))
    return torch.as_tensor(token_rows, dtype=torch.long, device=device)


@dataclass
class EvaluationSet:
    tokens: torch.Tensor
    labels: np.ndarray
    entity_indices: np.ndarray
    entity_count: int


def sample_negative(source: str, matched_ids: set[str], rng: random.Random, negative_pools):
    pool = negative_pools.get(source, ())
    if not pool:
        return None
    for _ in range(8):
        candidate = rng.choice(pool)
        if candidate[0] not in matched_ids:
            return candidate
    return None


def build_negative_pools(connection: sqlite3.Connection, pool_size: int, seed: int):
    rng = random.Random(seed)
    pools = {}
    for source in ("S2", "S3"):
        pool = []
        seen = 0
        cursor = connection.execute(
            "SELECT entity_id, name, address, country FROM records WHERE source = ?", (source,)
        )
        for row in cursor:
            seen += 1
            if len(pool) < pool_size:
                pool.append(row)
            else:
                index = rng.randrange(seen)
                if index < pool_size:
                    pool[index] = row
        pools[source] = pool
    return pools


def build_evaluation_set(
    connection: sqlite3.Connection,
    max_entities: int,
    seed: int,
    device: torch.device,
    negative_pools,
) -> EvaluationSet:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    cursor = connection.execute("SELECT entity_id FROM s1_split WHERE split = 'validation'")
    for (entity_id,) in cursor:
        seen += 1
        if len(reservoir) < max_entities:
            reservoir.append(entity_id)
        else:
            index = rng.randrange(seen)
            if index < max_entities:
                reservoir[index] = entity_id

    records: list[tuple[str, str, str, str, str, str]] = []
    labels: list[int] = []
    entity_indices: list[int] = []
    negative_rng = random.Random(seed + 1)
    for entity_index, s1_id in enumerate(reservoir):
        left_row = connection.execute(
            "SELECT name, address, country FROM records WHERE entity_id = ?", (s1_id,)
        ).fetchone()
        if left_row is None:
            continue
        positives = connection.execute(
            "SELECT r.entity_id, r.name, r.address, r.country FROM pairs p JOIN records r ON r.entity_id = p.target_id WHERE p.s1_id = ?",
            (s1_id,),
        ).fetchall()
        matched_ids = {row[0] for row in positives}
        for _, right_name, right_address, right_country in positives:
            records.append((*left_row, right_name, right_address, right_country))
            labels.append(1)
            entity_indices.append(entity_index)
        for source in ("S2", "S3"):
            negative = sample_negative(source, matched_ids, negative_rng, negative_pools)
            if negative is not None:
                records.append((*left_row, negative[1], negative[2], negative[3]))
                labels.append(0)
                entity_indices.append(entity_index)

    tokens = encode_batch(records, device) if records else torch.empty((0, 2, SEQUENCE_LENGTH), dtype=torch.long, device=device)
    return EvaluationSet(tokens, np.asarray(labels, dtype=np.int8), np.asarray(entity_indices, dtype=np.int32), len(reservoir))


def macro_f05(labels: np.ndarray, predictions: np.ndarray, entity_indices: np.ndarray, entity_count: int) -> float:
    if entity_count == 0:
        return 0.0
    true_positive = np.bincount(entity_indices, weights=(labels * predictions), minlength=entity_count)
    predicted_positive = np.bincount(entity_indices, weights=predictions, minlength=entity_count)
    actual_positive = np.bincount(entity_indices, weights=labels, minlength=entity_count)
    scores = np.ones(entity_count, dtype=np.float64)
    has_truth = actual_positive > 0
    scores[has_truth] = 0.0
    valid = has_truth & (predicted_positive > 0)
    precision = np.zeros(entity_count, dtype=np.float64)
    recall = np.zeros(entity_count, dtype=np.float64)
    precision[valid] = true_positive[valid] / predicted_positive[valid]
    recall[has_truth] = true_positive[has_truth] / actual_positive[has_truth]
    denominator = 0.25 * precision + recall
    scores[valid] = 1.25 * precision[valid] * recall[valid] / denominator[valid].clip(min=1e-12)
    return float(scores.mean())


def evaluate(model: nn.Module, evaluation: EvaluationSet, batch_size: int, device: torch.device):
    if len(evaluation.labels) == 0:
        return 0.0, 0.5, np.empty(0, dtype=np.float32)
    model.eval()
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(evaluation.labels), batch_size):
            logits = model(evaluation.tokens[start : start + batch_size])
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probability_batches)
    best_score, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(np.int8)
        score = macro_f05(evaluation.labels, predictions, evaluation.entity_indices, evaluation.entity_count)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_score, best_threshold, probabilities


def train(args) -> None:
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    logger.info("Device: %s", device)
    logger.info("Model parameters will remain below 200M; hashed vocabulary buckets: %s", f"{HASH_BUCKETS:,}")

    with tempfile.TemporaryDirectory(prefix="pair_model_cache_", dir=output_dir) as cache_dir:
        db_path = Path(cache_dir) / "records.sqlite"
        started = time.time()
        build_cache(Path(args.train_dir), db_path, args.validation_fraction, logger)
        connection = sqlite3.connect(db_path)
        try:
            negative_pools = build_negative_pools(connection, args.negative_pool_size, args.seed)
            logger.info(
                "Reservoir-sampled negative pools: S2=%s records, S3=%s records",
                f"{len(negative_pools['S2']):,}", f"{len(negative_pools['S3']):,}",
            )
            train_positive_count = connection.execute(
                "SELECT COUNT(*) FROM pairs p JOIN s1_split s ON s.entity_id = p.s1_id WHERE s.split = 'train'"
            ).fetchone()[0]
            validation = build_evaluation_set(
                connection, args.validation_entities, args.seed, device, negative_pools
            )
            logger.info(
                "Training positives: %s; validation entities sampled: %s; validation candidate pairs: %s",
                f"{train_positive_count:,}", f"{validation.entity_count:,}", f"{len(validation.labels):,}",
            )
            if train_positive_count == 0:
                raise ValueError("No training positive pairs found after the entity-level split.")

            model = PairEnergyModel().to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            logger.info("PairEnergyModel parameters: %s (%.2f MB float32)", f"{parameter_count:,}", parameter_count * 4 / 1_000_000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            loss_function = nn.BCEWithLogitsLoss()
            metrics_path = output_dir / "epoch_metrics.csv"
            best_f05 = -1.0
            best_threshold = 0.5
            with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
                metrics_writer = csv.writer(metrics_file)
                metrics_writer.writerow(("epoch", "steps", "mean_train_loss", "validation_macro_f0.5", "threshold", "seconds"))

                for epoch in range(1, args.epochs + 1):
                    model.train()
                    epoch_started = time.time()
                    batch_records: list[tuple[str, str, str, str, str, str]] = []
                    batch_labels: list[float] = []
                    running_loss = 0.0
                    steps = 0
                    positive_cursor = connection.execute(
                        """
                        SELECT a.name, a.address, a.country, b.name, b.address, b.country, p.s1_id, p.target_id, b.source
                        FROM pairs p
                        JOIN s1_split s ON s.entity_id = p.s1_id AND s.split = 'train'
                        JOIN records a ON a.entity_id = p.s1_id
                        JOIN records b ON b.entity_id = p.target_id
                        ORDER BY p.s1_id
                        """
                    )
                    stop_epoch = False
                    for s1_id, entity_rows in itertools.groupby(positive_cursor, key=lambda item: item[6]):
                        rows = list(entity_rows)
                        matched_ids = {row[7] for row in rows}
                        for row in rows:
                            left = row[:3]
                            batch_records.append((*left, *row[3:6]))
                            batch_labels.append(1.0)
                            negative = sample_negative(row[8], matched_ids, random, negative_pools)
                            if negative is not None:
                                batch_records.append((*left, negative[1], negative[2], negative[3]))
                                batch_labels.append(0.0)

                            if len(batch_labels) >= args.batch_size * 2:
                                batch_tokens = encode_batch(batch_records, device)
                                labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)
                                optimizer.zero_grad(set_to_none=True)
                                loss = loss_function(model(batch_tokens), labels)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                                running_loss += float(loss.item())
                                steps += 1
                                batch_records.clear()
                                batch_labels.clear()
                                if steps % args.log_every == 0:
                                    logger.info("epoch=%d step=%d train_loss=%.5f", epoch, steps, running_loss / steps)
                                if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                                    stop_epoch = True
                                    break
                        if stop_epoch:
                            break

                    if batch_labels:
                        batch_tokens = encode_batch(batch_records, device)
                        labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(model(batch_tokens), labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        running_loss += float(loss.item())
                        steps += 1

                    val_f05, threshold, probabilities = evaluate(model, validation, args.eval_batch_size, device)
                    elapsed = time.time() - epoch_started
                    mean_loss = running_loss / max(steps, 1)
                    logger.info(
                        "epoch=%d complete steps=%d train_loss=%.5f validation_macro_f0.5=%.5f threshold=%.2f seconds=%.1f",
                        epoch, steps, mean_loss, val_f05, threshold, elapsed,
                    )
                    metrics_writer.writerow((epoch, steps, f"{mean_loss:.7f}", f"{val_f05:.7f}", f"{threshold:.4f}", f"{elapsed:.1f}"))
                    metrics_file.flush()
                    if val_f05 > best_f05:
                        best_f05, best_threshold = val_f05, threshold
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "parameter_count": parameter_count,
                                "hash_buckets": HASH_BUCKETS,
                                "sequence_length": SEQUENCE_LENGTH,
                                "threshold": best_threshold,
                                "validation_macro_f0.5": best_f05,
                                "epoch": epoch,
                                "seed": args.seed,
                            },
                            output_dir / "best_pair_model.pt",
                        )
                        with (output_dir / "validation_scores.tsv").open("w", encoding="utf-8", newline="") as score_file:
                            writer = csv.writer(score_file, delimiter="\t")
                            writer.writerow(("entity_index", "label", "compatibility_probability"))
                            writer.writerows(
                                (int(entity_index), int(label), f"{float(probability):.7f}")
                                for entity_index, label, probability in zip(validation.entity_indices, validation.labels, probabilities)
                            )
                        logger.info("Saved new best checkpoint (macro F0.5 %.5f)", best_f05)
        finally:
            connection.close()

    summary = {
        "best_validation_macro_f0.5": best_f05,
        "best_threshold": best_threshold,
        "validation_entities_sampled": validation.entity_count,
        "validation_candidate_pairs": len(validation.labels),
        "validation_negatives_per_entity": "up to one per target source (S2 and S3)",
        "elapsed_seconds": round(time.time() - started, 2),
        "checkpoint": str(output_dir / "best_pair_model.pt"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    logger.info("Training finished: %s", json.dumps(summary, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a disk-backed pair-distribution model for entity resolution.")
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pair_model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512, help="Positive pairs per optimizer step; each gets a sampled negative.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-entities", type=int, default=10000)
    parser.add_argument("--negative-pool-size", type=int, default=50000, help="Reservoir size per target source; bounded in-memory negative sample.")
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Optional smoke-run cap; 0 streams all training positives.")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())