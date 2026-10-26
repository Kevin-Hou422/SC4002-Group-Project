"""Checks for data leakage, metric compatibility, and attention masking."""

import unittest
from collections import Counter
import json
import re

import torch

from gru_attention import (Config, GRUAttention, MassiveDataset, audit_data, build_vocabs,
                           collate, decode_bioes, parse_annotation, score_predictions, ROOT)


class GRUAttentionTests(unittest.TestCase):
    def test_bioes_roundtrip_with_duplicate_spans(self):
        text = "set [time : five am] and [time : five am] on [date : friday]"
        tokens, tags, slots = parse_annotation(text)
        self.assertEqual(tags, ["O", "B-time", "E-time", "O", "B-time", "E-time", "O", "S-date"])
        self.assertEqual(Counter(slots), Counter(decode_bioes(tokens, tags)))
        self.assertEqual(decode_bioes(["five", "am"], ["I-time", "E-time"]), [("time", "five am")])

    def test_exact_micro_f1_counts_duplicates_and_slot_types(self):
        rows = [{"gold_intent": "alarm", "pred_intent": "alarm",
                 "gold_slots": [("time", "five"), ("time", "five")],
                 "pred_slots": [("time", "five"), ("date", "five")]}]
        score = score_predictions(rows)
        self.assertEqual(score["intent_accuracy"], 100)
        self.assertEqual((score["slot_tp"], score["slot_fp"], score["slot_fn"]), (1, 1, 1))
        self.assertEqual(score["slot_micro_f1"], 50)
        self.assertEqual(score["joint_exact_match"], 0)

    def test_train_only_vocabulary_and_masked_padding(self):
        train = [{"id": "1", "utt": "wake me", "intent": "alarm", "annot_utt": "wake [person : me]"}]
        vocabs = build_vocabs(train)
        other = {**train[0], "id": "2", "utt": "unseen", "annot_utt": "unseen"}
        dataset = MassiveDataset(train + [other], vocabs)
        self.assertNotIn("unseen", vocabs["tokens"])
        self.assertEqual(dataset[1]["input_ids"].tolist(), [1])
        batch = collate(dataset.items)
        model = GRUAttention(len(vocabs["tokens"]), 1, len(vocabs["tags"]), Config(dropout=0)).eval()
        with torch.no_grad():
            intent, slots, weights = model(batch["input_ids"], batch["lengths"])
            alone = model(batch["input_ids"][1:2, :1], batch["lengths"][1:2])
        self.assertEqual(tuple(intent.shape), (2, 1))
        self.assertEqual(tuple(slots.shape[:2]), (2, 2))
        self.assertEqual(weights[1, 1].item(), 0)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        torch.testing.assert_close(intent[1], alone[0][0])
        torch.testing.assert_close(slots[1, 0], alone[1][0, 0])

    def test_full_official_split_and_annotation_integrity(self):
        audit, vocabs = audit_data(ROOT / "Data/data")
        self.assertEqual([audit[s]["examples"] for s in ("train", "dev", "test")], [11514, 2033, 2974])
        self.assertEqual(len(vocabs["intents"]), 60)
        self.assertEqual(len(vocabs["tags"]), 221)

    def test_scoring_and_decoder_match_existing_notebook(self):
        notebook = json.loads((ROOT / "Source/main.ipynb").read_text(encoding="utf-8"))
        namespace = {"Counter": Counter, "SLOT_REGEX": re.compile(r"\[([^\[\]:]+)\s*:\s*([^\[\]]+)\]")}
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and source.startswith(("def annot_to_bioes", "def decode_bioes", "def eval_intent_accuracy")):
                exec(source, namespace)
        tokens = ["a", "b", "c", "d", "e"]
        # Include mismatched, orphaned and unfinished BIOES sequences.
        for tags in (["B-x", "I-x", "E-x", "S-y", "O"],
                     ["I-x", "E-y", "B-x", "S-y", "B-x"]):
            self.assertEqual(decode_bioes(tokens, tags), namespace["decode_bioes"](tokens, tags))
        rows = [{"gold_intent": "a", "pred_intent": "a",
                 "gold_slots": [("x", "a"), ("x", "a")], "pred_slots": [("x", "a")]},
                {"gold_intent": "b", "pred_intent": "a", "gold_slots": [], "pred_slots": [("y", "b")]}]
        ours = score_predictions(rows)
        existing = namespace["evaluate"](*[[row[key] for row in rows]
                                          for key in ("gold_intent", "pred_intent", "gold_slots", "pred_slots")])
        self.assertEqual(round(ours["intent_accuracy"], 2), existing["intent_acc"])
        self.assertEqual(round(ours["slot_micro_f1"], 2), existing["slot_f1"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
