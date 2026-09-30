# Planner recovery: staged experiment

This experiment addresses Module 5: choosing the content of B's spoken reply from A's input. It does not assume that a smaller training loss means a more useful reply. The earlier acoustic experiments did not improve independent audio evaluation, so their candidate checkpoints are not promoted.

## Fixed setup

| Item | Value |
|---|---|
| Server project | `/home/aisha/bk-full-architecture` |
| Experiment directory | `outputs/planner_stages_v1` |
| Training conversations | 2,048, original training split |
| Development conversations | 128, original validation split |
| Repeated audio examples | Eight fixed development conversations after each epoch |
| Additional confirmation | 32 conversations outside those 128 and the earlier 24-case pilot |
| GPUs and batch | Four GPUs; 16 examples per GPU; global batch 64 |
| Epoch | 32 optimizer updates, one pass over all 2,048 examples |
| Conditional stage | Five epochs / 160 updates |
| Optimizer | AdamW, maximum LR 0.0001, warmup then cosine decay, gradient clipping 1 |
| Acoustic initializer | `outputs/quality_recovery/acoustic/best.pt` |
| Speech codebook | `outputs/broad_units_v1/codebook.pt`, 1,024 fixed units |
| Generation | Eight masked-unit refinement rounds, existing 32-step acoustic sampler |
| Style / voice group | Style 0 / voice-group 1; the actual synthesized voice identity is undocumented |

All original test audio and test metrics remain unused. The additional confirmation set is fresh for this series of detailed audio checks; older training used aggregate validation on the original validation split, so it is not a pristine final test set. Lexical filtering cannot detect every semantic paraphrase. `data_audit.json` records membership, hashes, and exclusions.

## 1. Conditional planner baseline

Initialize a new planner for the broader codebook. Keep the A encoders, fusion, affect transport, style/speaker embeddings, acoustic generator, and codec unchanged. Train only the planner and response-length predictor. Targets are the paired B recording's speech units and duration.

For example, A says, “I failed my exam.” Training asks the planner to predict the unit sequence belonging to that conversation's recorded B reply. At normal inference, B's recording, B's transcript, B's units, and B's duration are unavailable. Only A's measured inputs and the requested style/voice group enter generation.

| Boundary | Shape for a batch of size `B` |
|---|---|
| Fused A context | `[B, T_A, 512]` |
| A-conditioned target affect trajectory | `[B, T_A, 6]` |
| Requested style embedding | `[B, 256]` in this checkpoint configuration |
| Predicted response duration | `[B]`, seconds |
| B training semantic features | `[B, N_B, 768]`, approximately 50 frames/second |
| Quantized B target IDs | `[B, N_B]`, integers 0–1023 |
| Planner logits | `[B, N_B, 1024]` |
| Generated unit vectors | `[B, N_B, 768]`, fixed codebook lookup |
| Generated acoustic latents | `[B, T_B, 128]`, approximately 50 frames/second |
| Decoded waveform | `[B, samples]`, 32 kHz, variable valid length |

The implemented 50 Hz content-unit interface differs from the diagram's illustrative 12.5 Hz semantic rate. Shortening that sequence is a separate architectural question; it is not silently changed here.

## 2. Does A influence the planner usefully?

`evaluate_planner_controls.py` evaluates the same examples three ways: their correct A context, another conversation's A context, and zero context. Style and speaker stay fixed. It records fully masked cross entropy, sampled unit accuracy, changed output units, and duration error.

The unit diagnostic supplies the true B sequence length to isolate content prediction. This is explicitly an oracle-length diagnostic, not normal inference. Epoch audio generation separately predicts B's duration from A. Sensitivity to a different A is necessary, but changing nonsense into different nonsense is not understanding. Likewise, zero context is outside the training distribution and cannot alone establish whether A is ignored.

## 3. Actual generated responses after every epoch

Each epoch saves its own checkpoint and eight audio comparisons:

- The B reference recording.
- Codec reconstruction of that recording.
- Generated audio using B's correct quantized speech units and correct duration, isolating the acoustic path.
- Planner-generated units with the correct B duration, isolating content selection.
- Fully A-only generation with predicted duration, the actual inference path.

The output name `oracle_semantics` for a unit checkpoint means quantized B units, because `UnitSpeechSystem.generate_batch` applies the codebook. It is not a continuous-feature oracle in this experiment.

Whisper provides an independent transcript of each waveform. Reference word error rate measures agreement with the one recorded B reply; a different appropriate reply can have high WER. Therefore the review also examines whether the transcription is coherent and relevant to A. Automated transcription is imperfect and does not replace human listening. No human listening score or empathy score is invented.

Checkpoint `best.pt` means best validation objective, not best audio or best empathy. Every epoch candidate is preserved. Missing audio evaluations are replayed from the matching saved candidate if a run is resumed.

## 4. Conditional speech-unit prior comparison

If the baseline remains generic or incoherent, record that evidence in `baseline_review.json` and run the already authorized prior experiment:

1. Start a separate planner from the same frozen acoustic initializer and codebook.
2. For five epochs, hide 20–80% of B's speech units and reconstruct them from the visible units. Context, affect, and style are null. The prior objective never consumes A, and the duration predictor is frozen. The common dataset loader still reads the paired cache.
3. Keep that learned planner, restore ordinary A-to-B conditioning and duration training, and fine-tune for five epochs on the same 2,048 paired conversations.
4. Repeat the same eight audio examples and correct/shuffled/zero-A controls.

The prior teaches local speech-unit structure. It is not a guarantee of dialogue reasoning. Its partially masked validation loss is a different task from fully masked conditional validation and must not be compared numerically as if they were the same objective. This comparison also uses five extra training epochs; it tests whether the recipe helps, not a compute-matched causal claim about pretraining.

## 5. Architecture decision and confirmation

Only after comparing the conditional baseline and prior recipe, inspect whether repeated neighboring units make the content sequence unnecessarily long. A run-length audit measures how many tokens would remain if adjacent identical IDs were represented once, and checks that the original sequence is reconstructed exactly when the true run durations are retained.

Run-length compression does not produce a working shorter-sequence model by itself. Inference would need a trained duration expander and an appropriate target-length model. An oracle roundtrip does not demonstrate that predicted durations or speech quality will work. Record the measured benefit and decide whether that architectural experiment is warranted.

Choose the recipe/checkpoint on development evidence before opening the additional 32-case confirmation results. Confirmation should report actual A-only responses and the same oracle controls. Preserve failures as results; do not continue selecting checkpoints against the confirmation set.

## Commands and files

From the server project, start or safely resume the baseline with:

```bash
.venv/bin/python scripts/run_planner_stages.py --phase baseline
```

After a recorded baseline review supports the prior comparison:

```bash
.venv/bin/python scripts/run_planner_stages.py --phase prior
```

The controller uses one shared lock to prevent duplicate experiment controllers. `run_status.json`, stage logs, `metrics.jsonl`, saved epoch checkpoints, and `audio_step_*/report.json` preserve progress and evidence. Full training checkpoints retain optimizer and RNG state. The source code includes short English/Korean comments.

Verification: 45 relevant tests passed locally and on the server across the model, sampler, conditioning controls, prior, checkpoint audit, compression, and repetition checks. The checkpoint audits confirmed finite tensors and the intended frozen boundaries in all three training stages.

## Completed development results

The conditional baseline completed five epochs. The separate comparison completed five B-only prior epochs and five conditional fine-tuning epochs. Both conditional recipes saved and evaluated every epoch. Training used four GPUs and batch 16 per GPU throughout.

| Development measure | Conditional baseline | Prior + conditional fine-tuning |
|---|---:|---:|
| Fully masked unit CE, 128 cases, frame-weighted FP32 diagnostic | 6.266 | 6.049 |
| Correct-A sampled unit accuracy | 3.67% | 4.36% |
| Shuffled-A sampled unit accuracy | 2.66% | 2.78% |
| Zero-A sampled unit accuracy | 2.44% | 2.94% |
| Units changed when A is shuffled | 64.95% | 68.12% |
| Mean absolute predicted-duration error | 0.913 seconds | 0.913 seconds |
| Distinct generated units across 128 cases | 173 / 1,024 | 220 / 1,024 |
| Mean within-example adjacent repetition | 74.24% | 68.45% |
| Correct-B-unit oracle audio WER, eight fixed cases | 12.66% | 12.66% |
| A-only audio reference WER, same eight cases | 92.82% | 89.97% |

The small FP32 diagnostic/training-log differences are expected: the trainer validates with BF16 and averages per-batch CE, while the diagnostic scores examples and reports an explicit frame-weighted mean. Use like-for-like columns for comparison.

These results show a modest improvement in predicting recorded units, but **neither recipe produces reliable contextual replies**. In the final prior-assisted checkpoint:

- A is frustrated that plans to live with a brother failed. Generated ASR: “That sounds really cool.”
- A complains that a boyfriend keeps using the phone. Generated ASR: “That sounds really good.”
- A enjoys pictures of their children. Generated ASR: “Yes, it sounds really cool.” This can fit the context loosely, but the same positive opening across unrelated positive and negative situations does not demonstrate understanding.

The correct B-unit oracle preserves substantially more of the relevant reply. Correct response duration alone does not repair generated content. A influences the planner, but useful A-to-B content selection is weak. The data itself has common openings: 734 of 2,048 training replies begin “that sounds”; these openings are valid in many references, but the model often fails to continue into the specific content that distinguishes one conversation from another.

Repetition statistics are a diagnostic of the generated unit distribution, not an empathy score. Real validation responses use all 1,024 units and have 30.94% mean adjacent repetition. The model uses a narrower vocabulary with far more repetition. This supports output collapse as a current symptom; it does not uniquely identify one mathematical cause.

## Architecture assessment

The run-length audit covered all 2,048 training and 128 development conversations. Training targets shrink from 673,910 frames to 462,925 runs, a 31.31% reduction. Their effective rate becomes 34.40 Hz; development becomes 34.56 Hz. No selected example reaches 12.5 Hz by simply merging identical neighbors. Every sequence expands back exactly when supplied with its real run lengths.

Decision: retain the current interfaces for this comparison. A compact planner plus a learned duration expander remains a possible efficiency experiment, but these measurements do not show that it fixes the content problem. Forcing fourfold downsampling would not be the lossless transformation audited here. Before adopting shorter units, test its oracle acoustic quality, predicted length/durations, and actual A-only response relevance separately.

Five conditional epochs are a short pilot. The experiments do not establish convergence, an architectural ceiling, or the failure of larger-scale speech pretraining. No candidate is promoted and no long training run is automatically restarted.

## Confirmation on 32 additional conversations

Both fifth-epoch checkpoints were fixed in a hash-locked plan before these results were opened. They were chosen as the best development objective within each recipe, after reviewing all five epochs as inadequate. This confirmation is a comparative failure analysis; it does not turn a failed candidate into a release.

| Audio reconstruction/reference-agreement measure | Baseline | Prior + fine-tuning |
|---|---:|---:|
| Reference recording ASR WER | 2.83% | 2.83% |
| Codec reconstruction WER | 4.58% | 4.58% |
| Correct-B-unit oracle WER | 15.53% | 15.53% |
| A-only generation reference WER | 93.22% | 90.52% |

The modest free-reply WER decrease is **not evidence of better understanding**. The actual prior-assisted ASR transcripts still include:

- A reports a garage break-in and stolen equipment: “Yes, I'm really happy.”
- A feels anxious about presenting to the class: “That sounds really good.”
- A worries about a girlfriend's throat lump: “That sounds really funny. I think we need to know more about the world.”

There are occasional plausible generic acknowledgments. For disappointment with neighbors, the prior-assisted output is “That sounds really rough today. You get me.” This is a local improvement over the baseline's unrelated “Happy Halloween, sweetie,” but it does not offset the broad failure pattern. All 32 cases, rather than only these excerpts, are retained in the reports.

On these additional cases, correct-A unit accuracy is only 3.13% for the baseline and 3.47% after pretraining. For the prior-assisted model, shuffled A gives 3.28%, so useful conditioning remains weak. Predicted-duration mean absolute error is about 0.93 seconds for both. Generated confirmation waveforms contain no raw clipping; one prior-assisted **oracle-length** transcription reaches the ASR token limit, while none of the fully A-only transcriptions does. These waveform/transcription checks do not establish human-perceived quality.

**Final assessment:** the staged implementation and bounded experiments are complete. The code runs and the frozen boundaries are verified, but the model has not reached reliable, empathetic response generation. The observed bottleneck is weak content selection with restricted, repetitive predicted unit sequences; duration and the acoustic path have their own remaining errors. Keep these checkpoints as research results. A future experiment must improve content-bearing sequence learning and context use, and be judged with actual A-only audio—not just extend training because the loss decreases.

Server evidence: `outputs/planner_stages_v1`. Downloaded evidence: `outputs/server_training_results/planner_stages_v1`. Full optimizer/RNG checkpoints remain on the server; `inference_best.pt` exports contain the same model weights without optimizer state. `baseline_review.json`, `prior_review.json`, `architecture_review.json`, `confirmation/locked_plan.json`, and `final_result.json` record the decisions.
