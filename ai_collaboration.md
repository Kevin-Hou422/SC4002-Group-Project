# AI collaboration log: Stage 2

- **Tool:** Claude Opus 5.5
- **Scope:** Stage 2 diagnosis and experiments on the BiLSTM + attention baseline (`main_biLSTM_Attn.ipynb`)
- **Source:** full transcript copied from the shared conversation: [link](https://claude.ai/share/597eaaed-1b7c-4a4a-9431-4fa00c7dc120). 

---


**User:**

I'm working on a group project for the NLP course. The task is joint intent classification + slot filling on MASSIVE (en-US), 60 intents and 55 slot types.

There are some rules we need to follow:

no pretrained models except word vectors (we're using GloVe 300d)

max 20M params

train on train split only, dev for tuning/model selection, test only once at the very end

no extra data or augmentation

We have chosen biLSTM with attention as the baseline model. The goal now is to find the weaknesses and make improvements. Can you read through the notebook first and tell me your observations?

---

**AI:**

I’ll read through the notebook and come back with observations.

*[Tool activity: Ran 3 commands, read a file]*

I read the whole notebook, including the full training logs. The pipeline is mostly clean: the vocab comes from train only, the BIOES round-trip is checked, padding is masked in attention, and reloading the checkpoint reproduces the dev score. Below are the problems I found, roughly in order of importance.

1. Test-split usage needs fixing first (rule risk)

The baseline’s error analysis runs on test (cells 35–38: per-scenario accuracy, confusion matrix, per-slot F1, error samples). The attention section correctly says Stage 2 shouldn’t be guided by test errors, but the baseline section already was. Move those cells to dev.

final_evaluation() always scores test. Test has already been scored for both models, and with Stage 2 reusing this function it will be scored again for every variant. I’d add a flag so only dev is scored during development. Then run test once, at the end, on the final chosen system and report the Stage-1 rows from that same final run. The group should also decide how to explain the Stage-1 test numbers already in the notebook.

Minor: the tag vocab and slotTypes are built from all splits. The label set is fixed by the dataset, so it’s defensible, but building from train is cleaner. Notice that transport_descriptor has n=0 on dev and still gets F1 = 0 just from false positives.

2. Training setup: the learning rate is too high

In both logs the val loss is lowest around epochs 3–14 and then flat or rising, while train loss keeps dropping.

Nearly all the gain after about epoch 15 comes from ReduceLROnPlateau halving the LR, from 0.01 down to about 3e-4. The best epochs (62 and 69) are reached when the LR is already tiny.

From epoch ~50 to the best epoch, the dev score moves by only ~0.2. That’s noise-level, so the “best epoch” is partly chance.

Fix: start Adam at about 1e-3, possibly with a lower LR for the embeddings. You’ll likely get the same or a better result in far fewer epochs.

3. You compare single runs, one seed each

The attention gain is +0.9 intent and +1.9 slot F1 on test, from one seed each. Seed variance for BiLSTMs on MASSIVE can be around ±0.5–1 point. Every row in the report should be the mean ± std over 3–5 seeds, or the Stage-2 comparisons won’t hold up.

4. Slot decoding is the biggest weakness

Slots are picked by independent per-token argmax, and decode_bioes silently repairs invalid sequences. The dev errors show what this causes:

lunch meeting becomes two event_name spans.

desk lamp one becomes desk + lamp one.

city leeds becomes two place_name spans.

Truncated spans: two hours instead of two hours from now, rain instead of going to rain.

These are boundary and transition errors, which a CRF layer (or at least BIOES-constrained Viterbi decoding) is built to fix. A CRF adds only about 221² ≈ 49k parameters and uses no pretraining. It’s the most likely single win for slot F1.

5. Intent and slots barely talk to each other

The slot head gets one sentence-level context vector, and every token gets the same copy. That’s global information, not token-specific attention.

Nothing ties slot types to the intent, so the model can tag song_name inside a calendar_set utterance. Options:

a slot-gated mechanism (Goo et al. 2018)

feeding intent logits or an intent embedding into the slot head

per-token self-attention, so each token attends to the words relevant to it

The intent representation is attention-pooled only. Concatenating attention, max-pool and the final hidden states is a cheap thing to try.

6. OOV words and rare open-vocabulary slots

The weakest slot types are mostly free-text names: song_name 29, audiobook_name 34, podcast_name 21, playlist_name 42, music_descriptor 0. They’re rare and full of unseen words.

About 3.3% of dev tokens are <UNK>, because the vocab is train-only. Many of those words do have GloVe vectors, but the model never sees them.

Option A: a character-level CNN/BiLSTM feature, which is allowed and helps with names and morphology.

Option B: at inference, look up GloVe vectors for OOV words. This is legitimate because it uses no labels, only the pretrained vectors you’re allowed. It only works if the embeddings stay in GloVe space, so you’d freeze them or fine-tune through a projection. A larger GloVe vocab (say the top 30k words) also works, but watch the 20M cap: 50k × 300 is already 15M.

Lower-casing costs nothing here, since MASSIVE utterances are already lower-case.

7. Intent errors are partly label noise

Most intent confusions are general_quirky ↔ qa_factoid, qa_definition, news_query. general_quirky is a catch-all class in MASSIVE, so some of these errors are probably not fixable. I’d frame it that way in the report rather than chase it. The more fixable confusions are calendar_set ↔ calendar_query and play_music ↔ play_radio, where specific words like “set” or “station” decide the class.

8. Small code and reporting issues

eval_macro_f1 actually computes micro F1. The docstring says so, but rename the function so the report doesn’t get it wrong.

Slot F1 matches (type, span text) pairs and ignores position. That’s fine, but say so in the report, since MASSIVE’s official evaluation uses span-level F1 in seqeval style.

The loss weights intent and slot equally, and the slot loss is averaged over tokens dominated by O. A loss weight is a cheap knob to tune on dev.

What I’d do first: fix the test-split usage, then lower the LR and add multi-seed runs so the baseline numbers are trustworthy. After that, add a CRF, then char features or the OOV handling, then intent–slot interaction. Run each as an ablation on dev only.

I can start with the CRF plus the LR and seed changes in a copy of the notebook in your project folder.

---

**User:**

ok, let's first work on the learning rate. Use the same attention model as stage 1, only change the starting learning rate from 0.01 to 0.001. And do not modify any stage 1 cells, everything should go under the stage 2 section, save to a new checkpoint so the stage 1 one is not overwritten

---

**AI:**

I’ll add the Stage 2 cells to the notebook in your project folder. First, finding it:

*[Tool activity: Used 4 tools]*

Round-trip is byte-identical, so appending cells won’t touch any Stage 1 cell. Adding the Stage 2 cells now:

*[Tool activity: Ran a command on their computer]*

I added a Stage 2 section to Source/main_biLSTM_Attn.ipynb. I haven’t run it yet. I checked that all 55 Stage 1 cells are byte-for-byte unchanged, and that the new code parses.

What’s new (6 cells at the end):

Exp 1 training: the Stage 1 attention model with the same seed, scheduler, patience and checkpoint rule. The only change is lr=1e-3.

It saves to Results/stage2_attn_lr1e-3_best.pt.

An assert stops it if that path ever matches a Stage 1 checkpoint.

dev_evaluation(): reloads the best checkpoint and scores only the dev split, so Stage 2 never touches test. Results and history are saved to stage2_attn_lr1e-3_results.json and stage2_attn_lr1e-3_history.json.

Comparison table: Stage 1 vs Exp 1 on dev, with best epoch, epochs run, training time, intent accuracy, slot F1 and combined score.

The Stage 1 numbers are read from the saved JSON files, so this works after a kernel restart.

Overlay plots: val loss, train loss, dev combined score and the LR schedule, for both runs.

How to run it without touching Stage 1:

If the notebook is open in Jupyter, reload it from disk first. Otherwise saving your open copy will overwrite the new cells.

If your kernel is still live from the Stage 1 run, just run the new cells.

After a restart, run the setup cells and the definition cells (data loading, vocab, GloVe, model classes, training utilities, and the attention model cell).

Skip the two training cells and the two final_evaluation cells. Re-running the training cells would overwrite the Stage 1 checkpoints, and the final_evaluation cells score test.

Training should take about 10 minutes on CPU, like before. I’d guess the best epoch comes much earlier (say 20–40) and the dev score ends up similar or slightly better, but that’s for the run to show. Once you have the numbers, a 3-seed repeat of both runs would tell you whether the difference is real or seed noise.

---

**User:**

can you check the results of exp1 and give me your observations.

From my own observations, the best epoch is 38 instead of 69, so the training time is halved. the val loss is lower overall, but the shape is similar, min at epoch 14 and slowly rises, so the overfitting problem is not fixed.

---

**AI:**

*[Tool activity: Ran a command on their computer]*

Your read is right. The lower learning rate halved training time without fixing the overfitting. Here’s what the logs add.

Results on dev

|  | Stage 1 (lr 0.01) | Exp 1 (lr 0.001) | Δ |
| --- | --- | --- | --- |
| Intent acc | 86.42 | 86.82 | +0.40 |
| Slot F1 | 76.59 | 77.90 | +1.31 |
| Combined | 81.50 | 82.36 | +0.86 |
| Best epoch / epochs run | 69 / 84 | 38 / 53 |  |
| Training time | 619 s | 315 s | −49% |

1. The gain is mostly in slots, and the intent gain is within noise. On 2,033 dev utterances, +0.4 intent accuracy is 8 more correct predictions, which one seed can easily produce. The +1.3 slot F1 is more likely real, but we need 3 seeds to claim either.

2. It’s faster than “best epoch 38” suggests. Exp 1 hits 81.50, Stage 1’s final best, at epoch 23, before the LR is ever reduced. After about epoch 29 the combined score just wanders between 81.6 and 82.4. So epoch 38 is the luckiest point on a plateau, not a clear peak. The LR reductions add maybe 0.3–0.5 slot F1 and nothing for intent.

3. Almost all the overfitting comes from the intent head. The total loss hides this, so I split it by task:

|  | Val loss minimum | At the end (ep 53) | Rise |
| --- | --- | --- | --- |
| Intent | 0.541 (epoch 8) | 0.673 | +24% |
| Slot | 0.315 (epoch 18) | 0.345 | +10% |

Train intent loss falls to 0.12. Intent accuracy plateaus around epoch 9 (85.7–87.1 from then on), while slot F1 keeps rising until about epoch 45. The two tasks peak at different times, and the shared encoder keeps training for the slots while the intent head memorises.

4. The rising val loss is mostly overconfidence, not worse predictions. Intent loss rises 24% while intent accuracy stays flat. The model is getting more confident on the ~270 dev examples it gets wrong, not getting more of them wrong. So checkpointing on the metric is still sound. The rising loss is a calibration and generalisation-gap problem, and the ~1.0 dev loss vs ~0.24 train loss shows the gap is still there.

5. A likely cause: 61% of the parameters are fine-tuned embeddings. The embedding table is 5,304 × 300 ≈ 1.59M of the 2.59M parameters, trained on only 11.5k short utterances. That makes it easy to memorise the training words. It also shifts words seen in training away from GloVe space, while <UNK> and rare words don’t move, which hurts dev words the model hasn’t seen.

What I’d do next:

Keep lr = 1e-3 as the new default.

Run a few small regularisation experiments, changing one thing at a time:

Label smoothing on the intent loss (0.1). It targets the overconfidence directly and is a one-line change.

Freeze the embeddings, or give them about a tenth of the learning rate. This removes most of the trainable parameters.

Weight decay or higher dropout on the intent head. This tackles the head that’s overfitting.

After that, add the CRF. Slot F1 (77.9) has far more room to improve than intent (86.8), so that’s where the bigger gain is likely to be.

Should I add the label-smoothing and embedding experiments as Exp 2a and 2b? They’d follow the same structure as Exp 1: Stage 2 only, new checkpoints, dev only.

---

**User:**

ok go ahead with 2a and 2b first. for 2a, only apply smoothing in the training loss. keep the eval loss unsmoothed so val loss stays comparable with exp 1

---

**AI:**

I’ll add Exp 2a and 2b in the Stage 2 section. First, checking the current state of the notebook since you’ve run it:

*[Tool activity: Used 3 tools]*

Your computer’s shell here is a sandboxed Linux VM without PyTorch, so I’ll run the smoke test in my workspace instead, with the dataset copied over and random embeddings in place of GloVe:

*[Tool activity: Ran 3 commands, used 2 tools]*

While torch installs, I’m preparing the smoke-test script:

*[Tool activity: Used 2 tools, ran a command, read a file]*

The new cells are in the notebook. Torch is still installing for the smoke test, so I’ll post the result once it’s done.

*[Tool activity: Ran 3 commands, used 2 tools]*

I added Exp 2a and 2b to the Stage 2 section, and they ran end-to-end in a 2-epoch test. The 62 cells that were already there, including Exp 1, are byte-for-byte unchanged.

Exp 2a (intent label smoothing 0.1):

Smoothing applies only to the loss used for training. Validation still goes through the unchanged Stage 1 predict/compute_losses, so val loss is unsmoothed and directly comparable with Exp 1.

The logged train_intent_loss is unsmoothed too, so the train curves also line up with Exp 1. The smoothed loss that is actually trained on is logged separately as train_intent_loss_smoothed. In the test it was 2.51 smoothed vs 2.19 unsmoothed, as you’d expect, and the two were identical in 2b.

Exp 2b (embedding learning rate 0.1×, so 1e-4):

All other parameters stay at 1e-3.

Setting EXP2B_EMB_LR_SCALE = 0.0 freezes the embeddings instead, and the trainable-parameter count updates to match.

I chose 0.1× over freezing because about 7% of the vocabulary has no GloVe vector, and frozen, those words would keep their random starting vectors.

Shared pieces:

A train_model_s2 loop: the same scheduler, early stopping and checkpoint rule as Stage 1, plus those two options. It also prints val intent and slot loss separately each epoch.

New checkpoints and JSON files, named stage2_exp2a_* and stage2_exp2b_*. Only dev is scored.

A comparison table of Stage 1, Exp 1, 2a and 2b, plus overlay plots. The table shows the scores and, for each run, where val intent loss and val slot loss bottom out and how much they rise afterwards. That’s the overfitting signal you’re trying to fix.

To run: reload the notebook from disk, then run the setup and definition cells and the new Exp 2 cells. The Exp 2 cells now define their own helpers, so you don’t need to rerun Exp 1, and you should still skip the Stage 1 training and final_evaluation cells. Each run should take about 5–6 minutes on CPU, similar to Exp 1.

The test used random embeddings and only 2 epochs, so its scores don’t mean anything; it only shows that the code runs.

---

**User:**

is the intent overfitting actually fixed, or does label smoothing just make the loss look better by construction? and since the scores didn't really change, is it worth combining 2a + 2b, or should we move on to 3 CRF?

---

**AI:**

I’ll pull the 2a and 2b results and histories and compare them with Exp 1.

*[Tool activity: Ran a command on their computer]*

Short answers: label smoothing did more than make the loss look better, but it fixed overconfidence, not generalisation. Neither experiment moved the scores beyond seed noise. I’d skip the 2a + 2b combination, keep label smoothing, and move on to the CRF.

| Dev | Exp 1 | 2a (label smoothing) | 2b (emb lr ×0.1) |
| --- | --- | --- | --- |
| Intent / Slot F1 / Combined | 86.82 / 77.90 / 82.36 | 87.06 / 78.03 / 82.55 | 86.82 / 78.37 / 82.59 |
| Best epoch / epochs / time | 38 / 53 / 315 s | 43 / 58 / 331 s | 77 / 92 / 526 s |
| Val intent loss: min → end | 0.541 @8 → 0.673 (+24%) | 0.565 @13 → 0.593 (+5%) | 0.535 @16 → 0.622 (+16%) |
| Val slot loss: min → end | 0.315 @18 → 0.345 (+10%) | 0.304 @16 → 0.346 (+14%) | 0.299 @29 → 0.314 (+5%) |

Is the intent overfitting fixed?

The val-loss improvement is not by construction. You kept the val loss unsmoothed, so the drop from +24% to +5% is measured on the same unsmoothed loss as Exp 1. Only the smoothed training loss is higher by construction.

But the mechanism is limiting confidence, not learning to generalise better. Smoothing stops the logits from growing without bound, so the ~265 dev utterances the model gets wrong are no longer predicted with near-certainty. That is exactly what was driving the loss rise.

The number of mistakes barely changed. Intent accuracy moved by +0.24, which is about 5 utterances. The lowest val intent loss is actually slightly worse (0.565 vs 0.541), because the model can no longer be fully confident on the examples it gets right.

For the report, I’d say: “label smoothing removed the rise in validation intent loss without changing accuracy, which shows the rise was overconfidence rather than a loss of generalisation.”

If you want direct evidence, compare the average confidence on wrong dev predictions between Exp 1 and 2a. That’s a few lines in a dev-only cell.

2b (embedding lr ×0.1) mostly helped the slot side. It has the lowest val slot loss, the smallest slot-loss rise, and the best slot F1 (+0.47). But it trains much more slowly: it reaches 82.46 at epoch 53 and only gets to 82.59 when the learning rate is down to about 2e-5, at 1.7× the training time.

Combine 2a + 2b or move to the CRF? I’d move to the CRF:

The three combined scores sit within 0.25 of each other, a smaller spread than Exp 1’s own score moves epoch to epoch around its best checkpoint. A 2a + 2b run would almost certainly land in the same range and couldn’t be told apart without several seeds.

The CRF goes after the actual biggest weakness: the slot boundary and fragmentation errors.

Carry label smoothing forward. It costs nothing, it only touches the intent loss so it doesn’t interact with the CRF, and it gives you a clear calibration point for the report.

Drop 2b as a default and report it as a neutral result. Its benefit was on the slot loss, which the CRF replaces with its own loss anyway, and it nearly doubles training time.

With the deadline on the 13th, runs are better spent on the CRF and a few seeds of the final configuration than on combinations that differ by about 0.2.

Should I add Exp 3 as Exp 2a plus a CRF slot layer with BIOES-constrained Viterbi decoding? It would follow the same pattern: Stage 2 only, a new checkpoint, dev only.

---

**User:**

agree, move to exp 3 CRF, also, can you add the confidence-on-wrong-predictions check (exp 1 vs 2a, dev only) along with exp 3?

---

**AI:**

I’ll add both to the Stage 2 section: the confidence check first, then Exp 3. Checking the notebook’s current state:

*[Tool activity: Ran a command on their computer]*

Writing the new cells and testing them in my workspace before touching your notebook:

*[Tool activity: Created a file, ran a command]*

Now the smoke test (2 epochs, random embeddings), plus a brute-force check that the CRF’s likelihood and Viterbi are exactly right:

*[Tool activity: Ran 2 commands]*

80× faster and numerically identical. Rerunning the full smoke test:

*[Tool activity: Used 3 tools, ran 2 commands]*

I’ve added the confidence check and Exp 3 to the end of main_biLSTM_Attn.ipynb (13 new cells). I haven’t run them on your data yet. The 72 cells already there, including your 2a/2b outputs, are unchanged.

Confidence check (Exp 1 vs 2a, dev only): this reloads both checkpoints and compares the wrong intent predictions. It reports the number of errors, their mean and median confidence, the share above 0.9 confidence, the probability given to the gold intent, the mean loss, and calibration error, plus a histogram. If 2a’s mistakes show clearly lower confidence while the error count stays the same, that confirms the overconfidence explanation.

Exp 3 has two parts:

3a: constrained decoding only, no retraining. It re-decodes the Exp 2a checkpoint so that tag sequences breaking the BIOES rules are impossible. This shows how much comes from just forbidding invalid sequences.

3b: the full CRF layer, trained. This is Exp 2a plus a CRF layer that learns tag-to-tag transition scores (about 49k extra parameters), with the same BIOES rules.

Same seed and initialisation as 2a, and the same label smoothing.

It saves to the new checkpoint stage2_exp3b_crf_ls0.1_best.pt.

Comparison output:

Slot precision and recall, the number of predicted spans, utterances with at least one slot error, and how many raw tag sequences break BIOES. The last one shows how often decode_bioes has been patching things up.

Per-slot-type F1 for 2a vs 3b, sorted by the size of the change.

Example utterances the CRF fixed and ones it newly broke.

Overlay plots of the training curves.

How I checked it:

The CRF’s loss and Viterbi output match brute-force enumeration on a toy tag set.

No gold training sequence breaks the BIOES rules.

A 2-epoch end-to-end run (random embeddings, so the scores are meaningless) went through without errors.

My first version took 156 s per epoch because of how it computed the CRF loss. I rewrote that step and got the same values 80× faster, so an epoch now takes about 17 s against 14 s without the CRF. A full run should take roughly 6–8 minutes.

Two things to note in the report:

For 3b, val slot loss is the CRF’s loss per token, so it isn’t comparable with the earlier slot losses. Val intent loss is still comparable.

Expect a trade-off. The 2-epoch test already showed the CRF fixing broken spans (e.g. smart plug + socket → smart plug socket) while occasionally breaking a few correct ones.

To run: reload the notebook from disk, then run the setup and definition cells and the new cells. The new cells redefine every helper they need, so you don’t need to rerun Exp 1/2 or any test-split cell.

---
