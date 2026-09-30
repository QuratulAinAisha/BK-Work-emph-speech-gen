# Sequential planner diagnostics

The goal is to distinguish three possible contributors to bad B responses: decoding that reinforces repetition, A information that is hard to recover after encoding, and weak response-sequence learning. This investigation uses the saved prior-assisted planner checkpoint from `outputs/planner_stages_v1/prior_finetune/best.pt`. Production model weights and acoustic generation remain unchanged.

## Fixed data and interpretation

The diagnostic training set has 2,048 original-training conversations; the development set has 128 original-validation conversations. Eight previously selected development conversations are used for repeated audio comparisons. These are diagnostics on an already examined development set, not a new generalization claim. Original test audio remains unused. B targets are supplied only in explicitly labeled oracle/hint conditions.

Source recordings, unit sequences, and ASR transcriptions answer different questions. Exact reference WER is useful for reconstruction and transcription, but a different appropriate free reply can disagree with the recorded B sentence. None of these automated checks is a human listening or empathy score.

## Step 1: decoding, without training

The checkpoint is evaluated with one, eight and sixteen greedy refinement rounds; eight and sixteen rounds that can reopen accepted predictions; and eight-round categorical sampling with temperature 0.8, top-k 20, seeds 42 and 43. Correct/shuffled A controls cover 128 conversations at the true B length. Audio variants use the same A-predicted duration and the same acoustic noise seed, with a separate unit-sampling RNG.

The reference eight-round sampler reproduces production unit vectors in regression tests, and the real audio diagnostic asserts exact waveform equality with production inference. Supplied visible hints remain immutable. Revisable decoding is a bounded ablation: previously accepted tokens are scored while visible, so its confidence ranking does not test every possible revision method.

| Correct-A decoding | Recorded-unit accuracy | Mean adjacent repetition | Distinct units across 128 cases |
|---|---:|---:|---:|
| Greedy, one round | 4.48% | 65.69% | 223 |
| Greedy, eight rounds | 4.36% | 68.45% | 220 |
| Greedy, sixteen rounds | 4.31% | 69.65% | 218 |
| Revisable, eight rounds | 4.28% | 68.26% | 228 |
| Revisable, sixteen rounds | 4.21% | 71.19% | 214 |
| Categorical, seed 42 | 3.78% | 17.34% | 462 |
| Categorical, seed 43 | 3.76% | 19.44% | 463 |

All 56 generated ASR transcripts were inspected. None of the variants reliably recovers relevant response content. Sampling spreads predictions across more units but continues to produce inappropriate positives, fragments, and unrelated wording. An example about being scared for the weekend becomes “That sounds really cool” or “That sounds really fun” in the sampled variants. Revisable eight-round decoding produces an ASR repetition/token-limit failure in one case.

Decision: no sampler promoted. Repetition is already high on the first pass, so irreversible commitments cannot alone explain the failure. The bounded experiments do not prove that every possible decoding method would fail.

## Step 2: source A audio and recoverable words

First, independently transcribe the actual source A recording mapped through the same source index and mel-file stem used by preparation. Validate response-file/style/voice mapping, save file hashes, and handle long recordings using timestamp-aware overlapping ASR chunks. The frozen Whisper-base.en audit of 128 A recordings gives corpus WER **4.11%** and corpus CER **1.72%**, with zero empty recognitions or decoder-limit flags. All selected recordings are shorter than 30 seconds. Several larger mismatches involve contractions, spelling, or spoken versus written numbers. This does not indicate widespread A/transcript mismatching.

Next, freeze the response model and extract four representations:

| View | Feature width | Exact position |
|---|---:|---|
| Raw | 768 | Cached HuBERT A features |
| Projected | 512 | Learned projection, before positional encoding |
| Speech | 512 | After the two speech-context Transformer layers |
| Fused | 512 | Final context supplied to the response planner |

Train a new character-CTC reader for each view, using only the same 2,048 training transcripts. Each reader has hidden width 256, one Transformer layer, AdamW LR 0.001, batch 16, and an initial 20-epoch budget. All four were extended equally to 40 epochs to check whether the gap closed with more updates. Four independent readers ran on four GPUs. They do not update or replace production modules. Per-epoch train64 and validation128 CER/WER are retained, along with exact feature provenance, masks, and resumable reader checkpoints. No transcript or feature is truncated; all views passed CTC frame-feasibility checks.

The raw reader has a different input width, so its input-layer parameter count and random initialization stream differ from the 512-dimensional readers. Interpret differences with this limitation and the observed learning curves. A failed reader alone cannot prove that information is absent. Successful word recovery also cannot prove conversational understanding.

### Completed results and the timing control

The final context contains approximately half as many frames as the speech representation: mean validation lengths are 119.2 versus 240.6 frames. A fifth reader therefore uses **only speech features**, resampled with the exact production interpolation to each example's fused frame count. No fused values or B targets enter this control. Its cache is separate and the original cache is preserved.

| Fresh reader input | Approximate rate | WER after 20 epochs | WER after 40 epochs | CER after 40 epochs |
|---|---:|---:|---:|---:|
| Raw HuBERT | 50 Hz | 32.88% | 30.99% | 10.06% |
| Projected HuBERT | 50 Hz | 34.02% | 33.32% | 10.07% |
| Speech-context output | 50 Hz | 25.96% | 26.66% | 8.19% |
| Same speech, resampled to fusion lengths | About 25 Hz | 50.89% | 54.19% | 17.26% |
| Final fused context | About 25 Hz | 52.73% | 53.60% | 16.84% |

Lower WER/CER is better. For example, a small letter or spacing error can count as an incorrect word; a 54% WER does not mean that 54% of a sentence's meaning disappeared. These are fresh small readers, not Whisper. The existing production input-content reader gives 22.99% WER and 7.36% CER, but its different training history makes it an unmatched comparison.

**Interpretation:** reducing the speech frame count reproduces essentially the entire reader gap. The results do not support blaming the additive fusion operation itself. They justify testing whether Module 5 benefits from access to speech features at their original rate, while retaining the fused stream for affect and acoustic conditioning. An otherwise identical control should upsample the reduced speech features back to the original frame count, separating retained temporal detail from merely providing more positions.

This is evidence about lexical decoding under one reader architecture, seed, and data budget. CTC becomes harder with fewer frames even when every target remains mathematically feasible. More reader training did not close the validation gap, and speech features have a history of transcript supervision. None of this proves irreversible semantic loss or a cause of inappropriate B replies.

## Step 3: progressively remove B hints

After reviewing Step 2, we compared 25%, 50%, 75%, and 100% hidden B units with two deterministic nested mask seeds. We scored only hidden units, keeping visible correct units out of the accuracy denominator, and compared correct and shuffled A. We also rendered eight examples with 0%, 25%, 50%, 75%, and 100% hidden units, preserving supplied visible units throughout refinement.

Every hint condition uses the true B duration. Even the 100%-hidden audio is therefore an oracle-length diagnostic, not fully normal A-only generation. The first 20% of frames is compared with the remaining 80% as a temporal proxy; it is not a verified word boundary or exact measurement of a generic opening. Mask seeds are repeated measurements of the same cases, not additional independent examples.

### Completed hidden-unit results

These are first-pass predictions on 128 validation conversations, aggregated over two mask seeds. The 100%-hidden predictions are identical across mask seeds; they do not double the number of independent examples.

| B units hidden | Correct-A hidden-unit accuracy | Shuffled-A accuracy | Correct-A first 20% of time | Correct-A remaining 80% | Correct-A hidden-unit cross-entropy |
|---|---:|---:|---:|---:|---:|
| 25% | 6.02% | 5.33% | 22.06% | 1.89% | 5.912 |
| 50% | 5.94% | 5.02% | 21.57% | 2.00% | 5.932 |
| 75% | 5.62% | 4.44% | 20.08% | 1.99% | 5.950 |
| 100% | 4.48% | 2.98% | 14.77% | 1.88% | 6.049 |

Correct A helps, particularly near the beginning, but the later part remains very weak even with 75% of B units visible. This contradicts the more optimistic explanation that the planner can already reconstruct good speech and only needs help selecting relevant content. It shows weak completion of missing units as well as weak A-conditioned generation in this checkpoint. It does not prove that every later frame is conversation-specific or that every opening is generic. Changing the hidden fraction also changes the DiT's time conditioning, so this curve does not isolate the causal contribution of hints alone; that would require correct versus shuffled visible hints at the same mask.

### Completed audio results

| B units hidden | Mean ASR WER against recorded B, eight examples |
|---|---:|
| Original B recording | 0.69% |
| 0%: all correct units supplied | 12.66% |
| 25% | 79.19% |
| 50% | 83.99% |
| 75% | 89.14% |
| 100%: no B units, true B duration supplied | 92.87% |

All 40 generated waveforms preserved their visible hints, had zero measured clipping, and produced no ASR token-limit flags. Their ASR transcripts were reviewed; this was not a human listening assessment. WER is a reconstruction measure here, not an empathy score or an exhaustive measure of all valid free replies. Errors in even a minority of discrete units can disrupt the acoustic decoder, so 25% hidden does not imply at most 25% word errors.

Example: A says, “My boyfriend is always talking on the phone when we are hanging out.” With all correct B units, ASR reads “That sounds really frustrating. Like you're being ignored when you want to connect.” With 25% hidden, the opening remains but the continuation becomes “when you being able to make the world more convenient.” With all units hidden, the transcript is “That's how I'm looking.” These deliberately hinted outputs are not examples of successful normal inference.

The masking and target audit found no hard bug explaining this result: visible units enter the planner's input projection and self-attention; only padding is excluded; training and diagnostics quantize the same normalized B representation. All 1,024 saved codebook centers round-trip to their own IDs. Perturbing visible IDs changes hidden predictions in a numerical sensitivity check, so hints are not simply disconnected. This does not establish that the planner uses them effectively. The previous B-only warm-up comprised only 160 updates and ended with validation cross-entropy around 6.26; it should not be described as a strong speech prior.

## Decision after all three checks

No sampler or checkpoint is promoted. No production weights changed. The three checks are complete, and the current model still fails to generate reliable, context-appropriate responses.

The first isolated architecture experiment should give **Module 5 access to the original-rate A speech memory**. Keep the affect and acoustic paths fixed and compare against the same architecture receiving speech reduced to 25 Hz and then interpolated back to 50 Hz. Match initialization, training examples, update budget, and evaluation cases. This directly tests the lead identified by the timing control without assuming that better transcript recovery guarantees better replies.

Before scaling that change, require both improved hidden-unit completion and improved actual A-only audio on fixed development examples, with correct/shuffled A controls. Evaluate intelligibility and contextual appropriateness separately; a lower loss, more varied units, or lower single-reference WER alone is insufficient. Preserve a later unseen confirmation set and the original test split. An A–B contrastive objective remains a separate possible experiment, but this run does not establish that a semantic matching loss alone can repair the weak speech denoiser. Do not combine several new losses or restart the old 100-epoch run based on these diagnostic results.

## Reproduction

Run from `/home/aisha/bk-full-architecture`:

```bash
.venv/bin/python scripts/run_planner_diagnostics.py --stage decoding
.venv/bin/python scripts/run_planner_diagnostics.py --stage information
.venv/bin/python scripts/run_planner_diagnostics.py --stage information --probe-epochs 40
.venv/bin/python scripts/prepare_probe_timing_control.py --source outputs/planner_diagnostics_v1/step2_information --output outputs/planner_diagnostics_v1/step2_timing_control
.venv/bin/python scripts/check_a_information.py --checkpoint outputs/planner_stages_v1/prior_finetune/best.pt --manifest outputs/bk_quality_prepared/manifest.json --selection outputs/planner_stages_v1/selection.json --output outputs/planner_diagnostics_v1/step2_timing_control --train-view speech_resampled --epochs 40 --batch-size 16 --lr .001 --device cuda:0
.venv/bin/python scripts/check_a_information.py --checkpoint outputs/planner_stages_v1/prior_finetune/best.pt --manifest outputs/bk_quality_prepared/manifest.json --selection outputs/planner_stages_v1/selection.json --output outputs/planner_diagnostics_v1/step2_timing_control --summarize
.venv/bin/python scripts/run_planner_diagnostics.py --stage hints
```

Each transition requires a recorded review of the previous step. The controller uses the existing experiment lock. If the initial readers are still too undertrained to interpret, extending all four equally uses `--stage information --probe-epochs 40`; this is a diagnostic extension, not response-model training. Do not select a production sampler from repetition or WER alone.

Reports and diagnostic checkpoints are stored on the server in `outputs/planner_diagnostics_v1`. Local results, all compared audio examples, and five small reader checkpoints are copied to `outputs/server_training_results/planner_diagnostics_v1`; large feature caches stay on the server. The source and data hashes are recorded in the artifacts. Production checkpoint SHA-256 is `7d4f86662197ed128635801a6b1319639eef5801dfe177500746262b367c5bd3`.

New code contains short English/Korean comments. Twenty-three focused diagnostic tests passed locally and on the server. They cover production-sampler parity, fixed hints, deterministic masks and random-number isolation, hidden-only metric denominators, A-only feature boundaries, resumable readers, source-audio mapping, ASR budget handling, and exact timing interpolation with immutable source caches. The production checkpoint hash was rechecked after all diagnostics and remained unchanged. All four GPUs were idle at completion.
