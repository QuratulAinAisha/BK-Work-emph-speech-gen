# Staged recovery after quality-v2

The user authorized this recipe on September 29, 2026: diagnose the current model,
correct training mismatches, demonstrate a small controlled run, and only then
advance to broader training or a targeted discrete-unit planner redesign.

The original baseline remains stopped at epoch 87. The quality-v2 100-epoch run
was stopped with **epoch 32 saved**. Neither should be restarted automatically.
The preserved initialization is `outputs/quality_recovery/v2_initialization.pt`
on the server. All recovery outputs have their own directory.

**Completed experimental results:** see [RECOVERY_RESULTS.md](RECOVERY_RESULTS.md).
The discrete planner passed training-example reproduction (12.32% WER), but held-out
responses remain poor (94.43% at the final checkpoint). This is a successful
memorization diagnostic, not a completed general-purpose empathetic speech model.

## What the current-checkpoint diagnosis found

Six validation conversations, one per response style, were generated from the same
epoch-32 checkpoint. Whisper base.en was used consistently for this diagnostic.

| Path | Mean reference WER | Meaning |
|---|---:|---|
| Real B recording | 6.7% | The ASR control has some errors, including source pronunciation issues. |
| Real B codec features → frozen decoder | 8.6% | The codec/cache path largely preserves the words in these cases. |
| Real B semantics → learned audio generator → decoder | 31.3% | Acoustic generation also needs attention. |
| A → predicted semantics → audio, 4/8/24 steps | Approximately 100% or above | No useful response generation was demonstrated. |

WER may exceed 100% when the recognizer inserts many words. For a free response,
exact reference WER is only a diagnostic: a different sentence can be appropriate.
Here, inspection of the transcripts also showed generic or irrelevant output.
The original recording and oracle paths are reconstruction tests, where comparison
with a known transcript is appropriate. Six cases are not a population estimate.

Using the correct response length did not rescue the predicted-semantic path.
Changing only the sampling budget did not rescue it either. The semantic planner
is the leading bottleneck; acoustic degradation is a secondary concern.

## Corrected training design

| Change | Implementation |
|---|---|
| Stable conditions | Recovery phases are explicit, with separate optimizers and outputs. No automatic epoch-based phase transition. |
| Stable validation | Acoustic validation always uses reference semantics. Planner/joint validation always uses predicted semantics. |
| Learning rate | Fixed-step warmup and cosine decay, independent of the curriculum's changing total loss. |
| Best checkpoint | Compared only within the same phase and validation conditions; initialization is retained if subsequent validation is worse. |
| Sampling budget | Eight semantic sampling steps for both training and inference in the pilot. This is an aligned experimental choice, not a proven optimum. |
| Codec time range | Uniform sampling across 0–1, covering the near-noise and near-clean regions used at inference. |
| Frozen conditioning | The SBE, A adapter, style/voice embeddings, and affect modules remain frozen and in evaluation mode during the initial pilot. |
| Acoustic phase | Train codec generator and codec content head with reference semantics and reference content supervision. |
| Planner phase | Train semantic planner and length predictor with acoustic weights frozen. Include sampled semantic and downstream content supervision. |
| Future joint phase | Explicitly preserve reference-semantic examples; default candidate is 80% reference / 20% predicted. Not launched merely because an epoch count was reached. |
| Frozen audio teacher | Pinned wav2vec2-base-960h gives a differentiable transcript loss on decoded waveform samples. |

The pretrained audio teacher is independent of the trainable latent CTC heads.
Its parameters stay frozen, while gradients pass through its forward computation
and the frozen codec decoder into generated features. It is applied to one example
per GPU every four optimization steps in the pilot. This is additional supervision,
not proof that the resulting free response is appropriate.

Teacher: `facebook/wav2vec2-base-960h`, revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.

The installed Transformers/PyTorch combination initially failed to map legacy
position-convolution weight-normalization names. Recovery loads the checkpoint
strictly after translating those names, instead of accepting reinitialized weights.
Unused training-time speech masking is disabled. Validation on eight real pilot
recordings produced **5.4% mean reference WER**, finite nonzero waveform gradients,
and no teacher-weight gradients.

## Small-data experiment

The fixed selection contains **64 training conversations and 24 distinct validation
conversations**, all style 0 and voice group 1, with B audio between 3 and 8 seconds.
The original conversation-based splits remain unchanged. The final test split is
not used for development. Digits unsupported by the teacher alphabet are excluded
from this bounded pilot; extending training requires explicit normalization.

This is intentionally an easier diagnostic problem. It does not represent all
styles, voice groups, durations, or conversational situations.

On eight selected validation recordings before recovery:

| Path | Mean reference WER |
|---|---:|
| Original recording | 2.69% |
| Codec reconstruction | 1.25% |
| Audio generated from correct semantics | 14.06% |

These controls are easier than the six-style diagnosis above; compare before/after
using the same selected examples, not by mixing the two sets.

Training uses four GPUs with batch 16 per GPU (global 64). The acoustic pilot is
bounded at 400 optimizer steps and the subsequent planner pilot at 600. These are
small-data optimizer-step budgets, not another full-dataset 100-epoch run. Dataset
samples are kept in host memory to avoid repeatedly reading the large disk cache.

The 400-step acoustic pilot completed. Its selected checkpoint reduced matched
eight-example held-out oracle-semantic WER from **14.06% to 8.91%** (reference
recordings: 2.69%). All eight examples were within 20 percentage points of their
reference control, passing the declared acoustic development gate. Training-set
oracle-semantic WER on eight examples was 9.03%. The planner phase then began with
the acoustic generator frozen. These results do not establish A-to-B response quality.

The continuous planner completed 600 steps. The validation-selected checkpoint was
step 400. Its eight training responses had **206.16% mean reference WER**, and eight
validation responses had **167.49%**. Some ASR outputs repeated words until the token
limit. Reading the transcripts also showed irrelevant responses, so this was not just
penalizing valid alternative wording. The memorization gate failed. More continuous
planner training on the full dataset was not launched.

## Conditional discrete-unit experiment

The failed planner triggered the proposed Module 5 experiment. It is a separate
checkpoint architecture (`llm_free_speech_units_v1`) and separate output directory;
the preceding acoustic and continuous-planner checkpoints are preserved.

1. Fit 512 k-means centers to normalized cached B HuBERT features from the selected
   64 **training** conversations only. Validation and test speech are not used to
   fit the centers. Store normalization and manifest/selection identity with them.
2. Replace each 768-dimensional frame by its closest center ID. These are experimental
   local clusters of the cached final-layer features, not a pretrained phoneme vocabulary.
   Frame rate stays at 50 Hz. A center lookup restores the acoustic interface's 768D width.
3. Test correct B units through the existing acoustic generator. Before adaptation,
   eight validation examples had **17.17% WER**, versus **8.91%** for continuous B features
   on the same examples. This did not meet the stricter excess-error gate.
4. Adapt the acoustic generator for 400 steps to quantized reference features; evaluate
   the same acoustic gate before training a new planner.
5. If reconstruction passes, train a masked-unit planner for 1,200 bounded steps.
   Half the training sequences are fully masked, so the planner must learn to use A's
   context. The rest have random partial masks. Cross-entropy supervises unit IDs;
   the duration predictor has its existing regression loss. Acoustics and A encoding
   remain frozen. This initial unit-learning stage does not optimize ASR-head proxies.
6. Generate all response positions in parallel, refining the least confident masked
   positions for eight rounds. There is no LLM or intermediate text generation.
   The acoustic generator and frozen codec turn the resulting center sequence into audio.
7. Test training memorization using the latest checkpoint, and held-out output using
   the validation-selected best checkpoint. These checkpoint choices are explicitly
   recorded; they must not be presented as one matched model comparison.

| Interface | Dimensions |
|---|---|
| Person A fused context | `[batch, A time, 512]` |
| Module 3 affect trajectory | `[batch, A time, 6]` |
| Unit planner logits | `[batch, B frames, K unit classes]`; K=1,024 in the accepted interface |
| Predicted discrete IDs | `[batch, B frames]` |
| Center lookup for audio conditioning | `[batch, B frames, 768]` |
| Generated codec latents | `[batch, B frames, 128]` |
| Decoded audio | `[batch, samples at 32 kHz]` |

The unit controller is `scripts/run_unit_recovery.py`. It uses the same exclusive
recovery lock and four GPUs with batch 16 per GPU. Its actual GPU save/resume check
completed steps 2 then 4. New unit tests cover nearest-center assignment, frozen
components, A-context dependence, padding, target-free inference, and checkpoint
round-tripping. All 17 focused recovery/quality/unit tests passed locally.

The 512-unit adaptation failed: its selected checkpoint produced **33.04%** oracle
WER. It was not promoted to planner training. A single higher-resolution 1,024-unit
control with the **unchanged improved acoustic generator** achieved **11.41%** WER;
seven of eight cases were within 20 percentage points of their reference controls,
passing the same declared gate. This is a development-set choice, not an unbiased
test-set result. The frozen acoustic interface is saved under
`outputs/quality_recovery/units_1024/acoustic/best.pt`, explicitly marked **zero new
acoustic training steps**. Every non-planner tensor matches the improved continuous
acoustic checkpoint. The new 1,024-class planner then starts from fresh parameters.

`scripts/initialize_unit_interface.py` can reproduce this conversion after a matching,
hashed quantized-oracle report passes. The common loader supports both continuous
and unit checkpoints, including `infer_quality.py` for real A-only inference.

The design draws on [HuBERT's clustered targets](https://arxiv.org/abs/2106.07447),
[direct speech-to-speech discrete units](https://arxiv.org/abs/2107.05604), and
[parallel masked refinement](https://arxiv.org/abs/2202.04200). This implementation
is an experimental combination, not a reproduction of those systems or evidence that
their reported results transfer to empathetic dialogue.

The accepted unit planner completed 1,200 steps in about 5.2 minutes of training.
This timing is for repeatedly training the in-memory 64-example pilot, not a
full-dataset epoch estimate. Unit CE fell to about 0.009 while held-out CE increased;
the audio and unit diagnostics confirm overfitting. Its final checkpoint reproduced
all training unit frames when given reference length, and A-only audio on eight
training cases reached 12.32% WER. The validation-selected step-100 checkpoint and
final step-1,200 checkpoint both failed held-out response review. Joint/full-data
training was not automatically started after the memorization gate passed.

Reports and representative WAVs are copied locally under
`outputs/server_training_results/quality_recovery`. `units_1024_last.pt` and
`units_1024_best.pt` there are inference exports without optimizer/RNG state;
their SHA-256 hashes were verified after transfer. Full resumable training
checkpoints remain under the server experiment directories.

## Gates and interpretation

After acoustic training, eight training and eight validation examples are rendered
and transcribed. The automatic acoustic development gate requires:

- Mean reference-recording WER at most 20%.
- Mean generated oracle-semantic WER at most 20%.
- At most 10 percentage points of excess error over reference controls.
- At least 75% of examples within 20 percentage points of their own reference control.

These are declared engineering gates for this pilot, not published universal speech
quality thresholds or human ratings. Failure stops automatic stage advancement and
requires diagnosis; it does not trigger an unbounded retry loop.

After planner training, both train and validation examples are generated using
Person A inputs. A small-set memorization check requires at most 25% mean reference
WER on eight training examples with valid reference controls. This checks whether
the model can reproduce learned target responses. Held-out conversational relevance
must be reviewed separately; correct alternative responses cannot be rejected just
because their words differ from the dataset.

If planner memorization still fails, the next candidate is discrete speech-unit
prediction in Module 5. Before training such a planner, the quantized units must pass
an oracle reconstruction test through their acoustic interface. Token vocabulary,
frame rate, and acoustic conditioning must be compatible. This redesign is conditional
on evidence; it is not silently substituted for the current experiment.

## Commands and artifacts

Launch only when no recovery controller or smoke test is active:

```bash
.venv/bin/python -u scripts/run_unit_recovery.py --output outputs/quality_recovery/units_1024
```

This resumes or evaluates the existing bounded unit experiment on four GPUs with
batch 16 per GPU. It requires the prepared codebook and interface artifacts. It does
not extend a completed pilot into another 100-epoch run. The preceding continuous
experiment was run by `scripts/run_recovery.py`; it remains available for provenance.
Both controllers use the same exclusive lock. The initial continuous controller
verified shutdown of the old run and the waveform teacher, then ran a four-GPU
gradient preflight before the bounded stages.
Existing last checkpoints are resumed using identical data and recipe settings.
It does not automatically start full-data training after a proxy metric passes.

```text
outputs/quality_recovery/run_status.json
outputs/quality_recovery/diagnosis/report.json
outputs/quality_recovery/teacher_validation.json
outputs/quality_recovery/selection.json
outputs/quality_recovery/pilot_initial_val/report.json
outputs/quality_recovery/acoustic/{recipe.json,metrics.jsonl,last.pt,best.pt}
outputs/quality_recovery/acoustic_gate.json
outputs/quality_recovery/planner/{recipe.json,metrics.jsonl,last.pt,best.pt}
outputs/quality_recovery/planner_gate.json
outputs/quality_recovery/units_1024/{codebook.pt,acoustic_gate.json,planner_gate.json}
outputs/quality_recovery/units_1024/planner/{recipe.json,metrics.jsonl,last.pt,best.pt}
outputs/quality_recovery/units_1024/planner_train_last/report.json
outputs/quality_recovery/units_1024/planner_val_best/report.json
outputs/quality_recovery/units_1024/planner_val_last/report.json
outputs/quality_recovery/units_1024/unit_prediction_{last,best}.json
```

New implementation comments are short English/Korean explanations. Focused tests
cover full-range flow sampling, fixed validation conditions, reference anchors,
component freezing, waveform gradients, resampling, and gate behavior. The existing
quality inference test still verifies that normal generation cannot read B targets.

## What remains unproven

Intelligibility, naturalness, empathy, and open-ended conversational relevance remain
empirical goals. Male/female metadata still denotes unverified voice groups.
Valence/arousal/dominance still lack direct dataset labels. Pretrained speech
features and better optimization do not themselves create dialogue knowledge.

Normal generation uses only Person A's features and the requested style/voice.
Oracle semantic features and reference durations are used only in explicitly labeled
diagnostic paths. No human ratings are fabricated.
