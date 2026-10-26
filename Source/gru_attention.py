"""From-scratch GRU + attention baseline for MASSIVE.

Run from any directory: python Source/gru_attention.py train
Only utt is a model input; annotations provide supervision, never features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SLOT_REGEX = re.compile(r"\[([^\[\]:]+)\s*:\s*([^\[\]]+)\]")
EXPECTED_COUNTS = {"train": 11514, "dev": 2033, "test": 2974}
PAD, UNK, IGNORE = 0, 1, -100


@dataclass
class Config:
    seed: int = 42
    embedding_dim: int = 128
    hidden_size: int = 128  # per direction; a single bidirectional GRU layer
    attention_size: int = 128
    dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    max_epochs: int = 40
    patience: int = 8
    threads: int = 4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(data_dir, split):
    path = Path(data_dir) / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]}, got {len(rows)}")
    if any(row["partition"] != split or row["locale"] != "en-US" for row in rows):
        raise ValueError(f"Incorrect partition or locale in {path}")
    return rows


def parse_annotation(annot_utt):
    """Same whitespace tokens and BIOES convention as Source/main.ipynb."""
    tokens, tags, slots = [], [], []
    position = 0
    for match in SLOT_REGEX.finditer(annot_utt):
        outside = annot_utt[position:match.start()].split()
        tokens.extend(outside)
        tags.extend(["O"] * len(outside))
        kind, text = match.group(1).strip(), match.group(2).strip()
        words = text.split()
        if not words:
            raise ValueError("Empty annotated slot")
        tokens.extend(words)
        if len(words) == 1:
            tags.append(f"S-{kind}")
        else:
            tags.extend([f"B-{kind}"] + [f"I-{kind}"] * (len(words) - 2) + [f"E-{kind}"])
        slots.append((kind, text))
        position = match.end()
    outside = annot_utt[position:].split()
    tokens.extend(outside)
    tags.extend(["O"] * len(outside))
    return tokens, tags, slots


def decode_bioes(tokens, tags):
    """Use the existing notebook's deterministic repair of invalid BIOES runs."""
    if len(tokens) != len(tags):
        raise ValueError("Token/tag lengths differ")
    spans, current, words = [], None, []

    def close():
        nonlocal current, words
        if current is not None:
            spans.append((current, " ".join(words)))
        current, words = None, []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            close()
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "B":
            close()
            current, words = kind, [token]
        elif prefix in ("I", "E"):
            if current != kind:
                close()
                current = kind
            words.append(token)
            if prefix == "E":
                close()
        elif prefix == "S":
            close()
            spans.append((kind, token))
        else:
            close()
    close()
    return spans


def score_predictions(predictions):
    """Exact (type, surface-text) multiset matches, pooled across utterances.

    This matches main.ipynb, including duplicate slot pairs. Its eval_macro_f1
    function is actually micro F1; use the correct name here.
    """
    if not predictions:
        raise ValueError("Cannot score an empty split")
    correct = tp = fp = fn = joint = 0
    for row in predictions:
        gold, pred = Counter(map(tuple, row["gold_slots"])), Counter(map(tuple, row["pred_slots"]))
        match = sum((gold & pred).values())
        tp += match
        fp += sum(pred.values()) - match
        fn += sum(gold.values()) - match
        intent_ok = row["gold_intent"] == row["pred_intent"]
        correct += intent_ok
        joint += intent_ok and gold == pred
    precision = 100 * tp / (tp + fp) if tp + fp else 0.0
    recall = 100 * tp / (tp + fn) if tp + fn else 0.0
    return {
        "intent_accuracy": 100 * correct / len(predictions),
        "slot_micro_f1": 100 * 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "slot_precision": precision,
        "slot_recall": recall,
        "joint_exact_match": 100 * joint / len(predictions),
        "slot_tp": tp, "slot_fp": fp, "slot_fn": fn,
        "examples": len(predictions),
    }


def build_vocabs(train_rows):
    counts = Counter(token.lower() for row in train_rows for token in row["utt"].split())
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for token, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        vocab[token] = len(vocab)
    intents = sorted({row["intent"] for row in train_rows})
    slot_types = sorted({kind for row in train_rows for kind, _ in parse_annotation(row["annot_utt"])[2]})
    tags = ["O"] + [f"{prefix}-{kind}" for kind in slot_types for prefix in ("B", "I", "E", "S")]
    return {"tokens": vocab, "intents": intents, "tags": tags, "slot_types": slot_types}


def audit_data(data_dir):
    splits = {name: read_split(data_dir, name) for name in EXPECTED_COUNTS}
    seen = set()
    report = {}
    for name, rows in splits.items():
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"Repeated ID across dataset: {row['id']}")
            seen.add(row["id"])
            tokens, tags, slots = parse_annotation(row["annot_utt"])
            if not tokens or tokens != row["utt"].split():
                raise ValueError(f"Annotation/utt alignment failed: {name}/{row['id']}")
            if Counter(decode_bioes(tokens, tags)) != Counter(slots):
                raise ValueError(f"Lossy slot conversion: {name}/{row['id']}")
        report[name] = {"examples": len(rows), "sha256": sha256(Path(data_dir) / f"{name}.jsonl")}
    # Compare split records to the original source rather than trusting counts alone.
    source = Path(data_dir) / "en-US.jsonl"
    if source.exists():
        with source.open(encoding="utf-8") as handle:
            original = [json.loads(line) for line in handle if line.strip()]
        if len(original) != sum(EXPECTED_COUNTS.values()):
            raise ValueError("Original source count differs from split total")
        if {r["id"]: r for r in original} != {r["id"]: r for rows in splits.values() for r in rows}:
            raise ValueError("Split records differ from original en-US.jsonl")
    vocabs = build_vocabs(splits["train"])
    if len(vocabs["intents"]) != 60 or len(vocabs["slot_types"]) != 55:
        raise ValueError("Expected all 60 intents and 55 slot types in train")
    train_tokens = vocabs["tokens"]
    for name, rows in splits.items():
        words = [w.lower() for row in rows for w in row["utt"].split()]
        report[name]["oov_token_pct"] = 100 * sum(w not in train_tokens for w in words) / len(words)
    return report, vocabs


class MassiveDataset(Dataset):
    def __init__(self, rows, vocabs):
        self.items = []
        intent_ids = {name: idx for idx, name in enumerate(vocabs["intents"])}
        tag_ids = {name: idx for idx, name in enumerate(vocabs["tags"])}
        for row in rows:
            tokens = row["utt"].split()  # sole source of model inputs
            aligned, tags, slots = parse_annotation(row["annot_utt"])
            if tokens != aligned or not tokens:
                raise ValueError(f"Annotation alignment failed for {row['id']}")
            self.items.append({
                "input_ids": torch.tensor([vocabs["tokens"].get(t.lower(), UNK) for t in tokens]),
                "tag_ids": torch.tensor([tag_ids[t] for t in tags]),
                "intent_id": intent_ids[row["intent"]],
                "id": row["id"], "utt": row["utt"], "tokens": tokens,
                "gold_intent": row["intent"], "gold_slots": slots,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    return {
        "input_ids": pad_sequence([r["input_ids"] for r in items], batch_first=True, padding_value=PAD),
        "tag_ids": pad_sequence([r["tag_ids"] for r in items], batch_first=True, padding_value=IGNORE),
        "lengths": torch.tensor([len(r["tokens"]) for r in items]),
        "intent_ids": torch.tensor([r["intent_id"] for r in items]),
        "items": items,
    }


class GRUAttention(nn.Module):
    def __init__(self, vocab_size, intent_count, tag_count, config):
        super().__init__()
        width = 2 * config.hidden_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(width, config.attention_size), nn.Tanh(),
                                       nn.Linear(config.attention_size, 1, bias=False))
        self.intent_head = nn.Linear(width, intent_count)
        self.slot_head = nn.Sequential(nn.Linear(2 * width, config.hidden_size), nn.Tanh(),
                                       nn.Dropout(config.dropout), nn.Linear(config.hidden_size, tag_count))

    def forward(self, input_ids, lengths):
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.size(1))
        mask = torch.arange(input_ids.size(1), device=input_ids.device)[None, :] < lengths.to(input_ids.device)[:, None]
        energies = self.attention(hidden).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = energies.softmax(dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        intent_logits = self.intent_head(self.dropout(context))
        context_tokens = context.unsqueeze(1).expand(-1, hidden.size(1), -1)
        slot_logits = self.slot_head(self.dropout(torch.cat((hidden, context_tokens), dim=-1)))
        return intent_logits, slot_logits, weights


def make_loader(rows, vocabs, config, shuffle=False):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(MassiveDataset(rows, vocabs), batch_size=config.batch_size,
                      shuffle=shuffle, collate_fn=collate, num_workers=0, generator=generator)


def run_epoch(model, loader, device, vocabs, optimizer=None):
    training = optimizer is not None
    model.train(training)
    intent_sum = slot_sum = examples = tokens = 0
    predictions = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            gold_intent = batch["intent_ids"].to(device)
            gold_tags = batch["tag_ids"].to(device)
            intent_logits, slot_logits, _ = model(inputs, batch["lengths"])
            intent_loss = nn.functional.cross_entropy(intent_logits, gold_intent)
            slot_loss = nn.functional.cross_entropy(slot_logits.reshape(-1, slot_logits.size(-1)),
                                                     gold_tags.reshape(-1), ignore_index=IGNORE)
            loss = intent_loss + slot_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            count = len(batch["items"])
            token_count = int(batch["lengths"].sum())
            intent_sum += intent_loss.item() * count
            slot_sum += slot_loss.item() * token_count
            examples += count
            tokens += token_count
            if not training:
                intent_ids = intent_logits.argmax(-1).cpu().tolist()
                tag_ids = slot_logits.argmax(-1).cpu().tolist()
                for row, intent_id, predicted in zip(batch["items"], intent_ids, tag_ids):
                    tags = [vocabs["tags"][idx] for idx in predicted[:len(row["tokens"])]]
                    predictions.append({
                        "id": row["id"], "utt": row["utt"], "tokens": row["tokens"],
                        "gold_intent": row["gold_intent"], "pred_intent": vocabs["intents"][intent_id],
                        "gold_slots": row["gold_slots"], "pred_slots": decode_bioes(row["tokens"], tags),
                        "pred_tags": tags,
                    })
    losses = {"intent_loss": intent_sum / examples, "slot_loss": slot_sum / tokens}
    losses["loss"] = losses["intent_loss"] + losses["slot_loss"]
    if not training:
        losses.update(score_predictions(predictions))
    return losses, predictions


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config, vocabs = Config(**checkpoint["config"]), checkpoint["vocabs"]
    model = GRUAttention(len(vocabs["tokens"]), len(vocabs["intents"]), len(vocabs["tags"]), config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocabs, config, checkpoint


def predict(utterance, checkpoint_path):
    """Inference requires only raw utt and the saved checkpoint."""
    tokens = utterance.split()
    if not tokens:
        raise ValueError("Utterance cannot be blank")
    model, vocabs, config, _ = load_checkpoint(checkpoint_path)
    torch.set_num_threads(config.threads)
    inputs = torch.tensor([[vocabs["tokens"].get(t.lower(), UNK) for t in tokens]])
    with torch.inference_mode():
        intent, slots, attention = model(inputs, torch.tensor([len(tokens)]))
    tags = [vocabs["tags"][idx] for idx in slots[0].argmax(-1).tolist()]
    return {"utt": utterance, "intent": vocabs["intents"][intent.argmax(-1).item()],
            "slots": decode_bioes(tokens, tags), "tokens": tokens,
            "attention": attention[0].tolist()}


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train"]["loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["dev"]["loss"] for r in history], label="Validation")
    axes[0].set(ylabel="Intent CE + token slot CE", xlabel="Epoch", title="GRU + attention: loss")
    for key, label in (("intent_accuracy", "Intent accuracy"), ("slot_micro_f1", "Slot micro F1")):
        axes[1].plot(epochs, [r["dev"][key] for r in history], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Percent", title="Validation metrics")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "learning_curves.png", dpi=160)
    plt.close(fig)


def train(config=None, data_dir=None, output_dir=None):
    config = config or Config()
    data_dir = Path(data_dir or ROOT / "Data/data").resolve()
    output = Path(output_dir or ROOT / "Results/gru_attention/seed_42").resolve()
    # A second run must not overwrite a completed baseline or its test evaluation.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty; choose a fresh --output-dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    audit, vocabs = audit_data(data_dir)
    write_json(output / "data_audit.json", audit)
    write_json(output / "vocabs.json", vocabs)
    write_json(output / "config.json", asdict(config))
    train_loader = make_loader(read_split(data_dir, "train"), vocabs, config, shuffle=True)
    dev_loader = make_loader(read_split(data_dir, "dev"), vocabs, config)
    model = GRUAttention(len(vocabs["tokens"]), len(vocabs["intents"]), len(vocabs["tags"]), config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 20_000_000:
        raise ValueError(f"Parameter budget exceeded: {parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "platform": platform.platform(), "device": str(device), "threads": config.threads,
        "git_base_commit": commit, "source_sha256": sha256(__file__),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "environment.json", environment)
    checkpoint_path = output / "best.pt"
    history, best_score, best_epoch = [], -float("inf"), 0
    started = time.perf_counter()
    log_path = output / "training.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"GRU + attention baseline | base {commit} | device {device} | seed {config.seed}")
    log(f"Splits: {EXPECTED_COUNTS} | vocab {len(vocabs['tokens'])} | intents 60 | tags {len(vocabs['tags'])}")
    log(f"Parameters: {parameters:,} / 20,000,000 | selection: mean(dev intent accuracy, dev slot micro F1)")
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_metrics, _ = run_epoch(model, train_loader, device, vocabs, optimizer)
        dev_metrics, _ = run_epoch(model, dev_loader, device, vocabs)
        score = (dev_metrics["intent_accuracy"] + dev_metrics["slot_micro_f1"]) / 2
        history.append({"epoch": epoch, "learning_rate": lr, "train": train_metrics, "dev": dev_metrics,
                        "selection_score": score, "epoch_seconds": time.perf_counter() - epoch_start,
                        "elapsed_seconds": time.perf_counter() - started})
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "vocabs": vocabs,
                        "epoch": epoch, "selection_score": score, "dev_metrics": dev_metrics,
                        "parameter_count": parameters, "environment": environment}, checkpoint_path)
        write_json(output / "history.json", history)
        log(f"epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} | dev loss {dev_metrics['loss']:.4f} | "
            f"dev intent {dev_metrics['intent_accuracy']:.2f}% | dev slot F1 {dev_metrics['slot_micro_f1']:.2f}% | "
            f"mean {score:.2f} | lr {lr:g} | {history[-1]['epoch_seconds']:.1f}s" + (" | saved best" if improved else ""))
        scheduler.step(score)
        if epoch - best_epoch >= config.patience:
            log(f"Early stopping: {config.patience} epochs without validation improvement")
            break
    # Freeze selection before loading the test split for final evaluation.
    model, _, _, checkpoint = load_checkpoint(checkpoint_path, device)
    dev_metrics, dev_predictions = run_epoch(model, dev_loader, device, vocabs)
    for key in ("intent_accuracy", "slot_micro_f1", "loss"):
        if abs(dev_metrics[key] - checkpoint["dev_metrics"][key]) > 1e-6:
            raise AssertionError(f"Reloaded checkpoint validation mismatch: {key}")
    test_loader = make_loader(read_split(data_dir, "test"), vocabs, config)
    test_metrics, test_predictions = run_epoch(model, test_loader, device, vocabs)
    for name, predictions in (("dev", dev_predictions), ("test", test_predictions)):
        with (output / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "model": "BiGRU + masked additive attention (joint intent and BIOES slots)",
        "parameter_count": parameters, "trainable_parameter_count": parameters,
        "best_epoch": best_epoch, "epochs_trained": len(history), "seed": config.seed,
        "selection_metric": "arithmetic mean of dev intent accuracy and dev slot micro F1 (percent)",
        "best_selection_score": best_score, "checkpoint": "best.pt", "checkpoint_sha256": sha256(checkpoint_path),
        "training_seconds": history[-1]["elapsed_seconds"], "total_seconds": time.perf_counter() - started,
        "dev": dev_metrics, "test": test_metrics, "environment": environment,
        "test_evaluations_during_selection": 0,
    }
    write_json(output / "metrics.json", result)
    plot_history(history, output)
    lines = ["# GRU + attention baseline", "",
             f"Base commit: `{commit}`. Seed: {config.seed}. CPU. Parameters: {parameters:,}.",
             f"Selected epoch {best_epoch} of {len(history)} by mean validation intent accuracy / slot micro F1.", "",
             "| Split | Intent accuracy (%) | Slot micro F1 (%) |", "| --- | ---: | ---: |"]
    for name, metrics in (("Validation", dev_metrics), ("Test", test_metrics)):
        lines.append(f"| {name} | {metrics['intent_accuracy']:.2f} | {metrics['slot_micro_f1']:.2f} |")
    lines += ["", "The test split was evaluated only after checkpoint selection. Vocabularies use train only.",
              "Slot scoring uses exact (type, span-text) multiset matches, identical to Source/main.ipynb.",
              "All metrics in metrics.json retain full precision; table values are percentages rounded to two decimals.",
              "", f"Training time: {result['training_seconds']:.1f} seconds.",
              "", "Artifacts: best.pt, config.json, vocabs.json, data_audit.json, environment.json, history.json,",
              "training.log, dev_predictions.jsonl, test_predictions.jsonl, metrics.json, learning_curves.png."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Best epoch {best_epoch} | score {best_score:.2f} | checkpoint {checkpoint_path}")
    log(f"Final validation: intent {dev_metrics['intent_accuracy']:.2f}% | slot micro F1 {dev_metrics['slot_micro_f1']:.2f}%")
    log(f"Final test: intent {test_metrics['intent_accuracy']:.2f}% | slot micro F1 {test_metrics['slot_micro_f1']:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "audit", "predict"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Results/gru_attention/seed_42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--text")
    for field in ("seed", "max_epochs", "patience", "threads", "batch_size"):
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(Config(), field))
    args = parser.parse_args()
    if args.command == "audit":
        audit, vocabs = audit_data(args.data_dir)
        print(json.dumps({"splits": audit, "vocab_size": len(vocabs["tokens"]),
                          "intents": len(vocabs["intents"]), "slot_types": len(vocabs["slot_types"])}, indent=2))
    elif args.command == "train":
        config = Config(**{field: getattr(args, field) for field in ("seed", "max_epochs", "patience", "threads", "batch_size")})
        if min(config.max_epochs, config.patience, config.threads, config.batch_size) < 1:
            parser.error("epochs, patience, threads and batch size must be positive")
        train(config, args.data_dir, args.output_dir)
    else:
        if not args.text:
            parser.error("predict requires --text")
        print(json.dumps(predict(args.text, args.checkpoint or args.output_dir / "best.pt"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
