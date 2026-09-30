# Original-rate speech memory experiment

The previous diagnostics found that reducing A's speech representation from about 50 Hz to 25 Hz made words harder for a small reader to recover. This experiment tests whether preserving those speech features helps Module 5 produce better replies. It does not assume that this change solves the separate weakness in predicting missing B units.

## Locked comparison

| Arm | Module 5 context | Width | Timeline | Purpose |
|---|---|---:|---|---|
| `fused` | Existing fused context | 512 | About 25 Hz | Control for additional training |
| `resampled_speech` | A speech features reduced to fusion lengths, then expanded back | 512 | Original speech frame count | Same-length control with reduced temporal detail |
| `native_speech` | Original A speech-context features | 512 | Original speech frame count, about 50 Hz | Candidate preserving temporal detail |

The primary contrast is native versus resampled speech. Their architecture, parameter count, memory lengths, initialization, data, update budget, and optimizer settings match. Comparing either with fused context additionally changes its representation and length, so that comparison cannot isolate temporal detail alone.

Only the existing Module 5 parameters train. No new parameter tensors are added. The A encoder, fusion, affect transport, duration predictor, style/voice embeddings, codebook, and acoustic generator remain frozen. Module 3 and the acoustic generator still receive their original fused context. The planner gets the original affect trajectory with its own original mask, so changing speech-memory length does not inadvertently change affect interpolation.

All three arms start from the exact saved `outputs/planner_stages_v1/prior_finetune/best.pt` tensor state, SHA-256 `7d4f86662197ed128635801a6b1319639eef5801dfe177500746262b367c5bd3`. Each uses a fresh AdamW optimizer, LR 0.0001, the same warm-up/cosine schedule, seed 42, four GPUs and batch 16 per GPU (global batch 64). There are 2,048 training conversations, 32 optimizer updates per epoch, and **320 updates / 10 epochs per arm**. Native/resampled tensors have equal shapes and therefore matching dropout draw shapes; fused is an additional continuation control, not that exact shape-matched contrast.

The objective remains unit cross-entropy plus the existing duration-loss term. Duration loss is constant with respect to the trainable planner because the duration predictor and all its inputs are frozen. No new content, ASR, or contrastive loss is mixed into this comparison.

## Evaluation and interpretation

Before updates, check a real four-GPU backward pass for each arm and render eight fixed development examples. After every 64 updates (two epochs), save a checkpoint and render those same examples. Each endpoint is **`step_0320.pt`**, chosen before observing results. Do not select different checkpoints by minimum loss or by attractive individual audio samples.

At the final endpoint, compare correct, shuffled, and zero A context; hidden B-unit reconstruction; and real A-only audio. B-unit diagnostics use the true B length and are explicitly oracle-length tests. Normal predicted-length audio receives only A and requested style/voice. Visible B hints are diagnostic only. A new same-mask control permutes the visible B hints while preserving their histogram and the hidden fraction, allowing a cleaner test of useful hint order than changing the hidden fraction alone.

Reserve 32 further validation conversations before training. Exclude prior development/pilot/confirmation conversation IDs and lexical near-duplicates of those groups or original training inputs. This is fresh detailed confirmation, not a pristine final test: older runs already exposed aggregate validation information from the split. Original test audio and metrics remain untouched. Lock all three final checkpoint hashes before opening confirmation; never reselect candidates afterward.

Evaluate intelligibility and contextual appropriateness separately. Reference WER is not an empathy score: a valid free reply may differ from the recorded response, and ASR may hallucinate. Lower unit loss, lower repetition, or a better transcript reader alone does not establish a working response model. No candidate is promoted automatically.

## Running and outputs

On the server, from `/home/aisha/bk-full-architecture`:

```bash
.venv/bin/python scripts/prepare_planner_memory_confirmation.py --manifest outputs/bk_quality_prepared/manifest.json --selection outputs/planner_stages_v1/selection.json --pilot outputs/quality_recovery/selection.json --previous-confirmation outputs/planner_stages_v1/confirmation_selection.json --output outputs/planner_memory_v1 --count 32
.venv/bin/python scripts/run_planner_memory.py --phase train
# Review all endpoints and write development_review.json first. / 모든 결과를 검토한 후 검토 기록을 저장합니다.
.venv/bin/python scripts/run_planner_memory.py --phase confirmation
```

The controller preserves the shared experiment lock, immutable plan/source hashes, resumable training states, and candidate checkpoints. Every saved candidate receives a bitwise audit permitting changes only under `semantic_planner`; the codebook must remain identical. Results live in `outputs/planner_memory_v1`, with local copies under `outputs/server_training_results/planner_memory_v1`.

## Completed results

All three arms completed their planned 320 updates on four GPUs, with batch 16 per GPU. The common preflight passed on every rank. All 15 saved-candidate audits passed: 73 planner tensors changed, 329 shared tensors stayed identical, the codebook stayed fixed, and all tensors remained finite. All modes had exactly the same validation duration loss, 0.017015884. The original initializer is preserved.

| Fixed final checkpoint | Development unit accuracy, 128 cases | Confirmation unit accuracy, 32 cases | Hidden-tail accuracy with 75% B hints, development | Same hidden-tail measure, 64 training cases |
|---|---:|---:|---:|---:|
| Fused continuation | 5.228% | 4.648% | 3.552% | 4.259% |
| Reduced then restored speech | 5.230% | 4.648% | 3.383% | 4.078% |
| Original-rate speech | 5.204% | 4.696% | 3.395% | 4.102% |

Unit accuracy uses eight-round sampling and the true B length. Hidden-tail accuracy uses first-pass logits on the masked positions in the last 80% of response time; visible correct units are excluded. It is a temporal region, not a word-aligned measure. These numbers are not percentages of appropriate conversations.

The native-minus-resampled difference in development unit accuracy is **−0.0265 percentage points**, with paired 95% interval **[−0.1055, +0.0554]**. On confirmation it is **+0.0478 points**, interval **[−0.1779, +0.2488]**. Development hidden-tail difference is only **+0.0121 points**, interval **[−0.0672, +0.0896]**. The intervals use 10,000 conversation-level bootstrap resamples, seed 42. Both mask seeds are combined within each conversation. They describe case variability for these trained checkpoints, not uncertainty across independent training seeds, and are exploratory without multiple-comparison correction.

**Decision: native 50 Hz speech memory did not demonstrate a useful advantage in this pilot.** This does not invalidate the earlier word-reader result or prove that temporal detail never helps; it shows that this specific planner, initialization and 10-epoch budget did not turn that information into better replies.

### Actual generated audio

| Arm | Development A-only mean / median reference WER | Development ASR-limit flags | Confirmation A-only mean / median reference WER | Confirmation ASR-limit flags |
|---|---:|---:|---:|---:|
| Fused | 190.24% / 96.67% | 1 of 8 | 129.75% / 92.86% | 3 of 32 |
| Resampled | 155.56% / 90.56% | 1 of 8 | 109.25% / 88.56% | 1 of 32 |
| Native | 87.35% / 88.26% | 0 of 8 | 110.59% / 89.68% | 2 of 32 |

WER can exceed 100% when ASR inserts many words. Long repeated ASR strings on short waveforms are failure flags, not proof that every transcribed word was spoken. The large development mean-WER difference is dominated by such outliers. On the same seven uncapped native/resampled development cases, means are 86.64% and 88.77%. These values still cannot establish response appropriateness.

The 32-case confirmation preserves the same failure pattern. For the native model, A describes loneliness in college and ASR transcribes B as “That sounds really good.” A describes a successful piano recital and B is transcribed as “That must have been really frustrating.” Some frustration-related inputs receive a suitable opening, but detailed continuations remain absent or incoherent. These are transcript reviews, not human listening scores. Original B-unit controls are substantially more intelligible: their confirmation reference WER is 10.39% for every arm. They deliberately receive B targets and are not normal generation.

### What the B-hint controls add

Correctly ordered visible B hints help all arms compared with permuting exactly those same visible hints. The hidden mask, fraction/time condition and hint histogram remain fixed, and the 100%-hidden negative control is identical. The model is therefore not completely ignoring hints. Nevertheless, both training and validation gap reconstruction remain poor.

After the primary comparison, an explicitly **post-hoc** sanity check copied the closest visible B-unit ID into each missing position, choosing the left neighbor on ties. The predictor receives only visible positions and IDs. Hidden targets are used solely for scoring. All 768 partial-mask hashes match the original diagnostics.

| B units hidden | Nearest-visible copy: hidden-tail accuracy | Native learned planner: hidden-tail accuracy |
|---|---:|---:|
| 25% | 30.72% | 3.40% |
| 50% | 26.20% | 3.59% |
| 75% | 17.67% | 3.51% |

This is a strong reason to investigate local speech-unit reconstruction before adding another global content loss. The copy baseline is **not a response generator**: it requires B hints and cannot operate when every B unit is unknown. No copying shortcut has been installed in normal inference. This exploratory check was added after seeing results and is not a new preregistered primary metric.

## Final disposition

No candidate is promoted and no long training run is restarted. The feature-memory modes are available as opt-in research settings; existing checkpoints default to fused memory. The next bounded development target should establish competent speech-unit denoising, including beating the simple gap-copy baseline on held-out examples, before claiming that further A–B relevance losses or scaling will yield good replies. Success at hinted reconstruction would still need a separate A-only generation test.

Seventy-eight relevant automated tests passed locally and on the server. Audio produced during training was rendered from `last.pt` at step 320; each `endpoint_identity.json` verifies that all 402 model tensors, configuration, recipe and step match the fixed `step_0320.pt`, binding both checkpoint hashes and the audio report hash. No checkpoint was reselected because of an audio result. Full resumable checkpoints remain on the server; reports, audio examples and inference exports are retained locally.

The fixed-endpoint summary is generated with:

```bash
.venv/bin/python scripts/summarize_planner_memory.py --root outputs/planner_memory_v1 --attest-audio-alias fused
.venv/bin/python scripts/summarize_planner_memory.py --root outputs/planner_memory_v1 --attest-audio-alias resampled_speech
.venv/bin/python scripts/summarize_planner_memory.py --root outputs/planner_memory_v1 --attest-audio-alias native_speech
.venv/bin/python scripts/summarize_planner_memory.py --root outputs/planner_memory_v1 --output outputs/planner_memory_v1/summary.json
.venv/bin/python outputs/local_analysis/nearest_visible_copy.py --root outputs/planner_memory_v1
```
