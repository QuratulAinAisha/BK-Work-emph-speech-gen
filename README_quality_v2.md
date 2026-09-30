# Speech quality improvements: version 2

**Current experiment:** the user authorized staged recovery on September 29, 2026.
This version-2 run was stopped with epoch 32 saved. See
[README_quality_recovery.md](README_quality_recovery.md) for the diagnosis, corrected
training, and gates. Do not restart the old 100-epoch controller automatically.

The goal is understandable, relevant Person B speech while keeping the system LLM-free.
Working tensor shapes and decreasing losses do not establish that goal. The baseline
code and checkpoints are preserved. Version 2 has separate data, checkpoints,
training logs, and evaluations.

On September 28, 2026, the user requested stopping the baseline if its latest
generated example remained inadequate. A snapshot of the latest completed checkpoint
at the start of that check (epoch 86) produced a 6.77-second response to an excited
movie-related input. Whisper tiny.en recognized only “you”; base.en recognized “I'm
sorry.” Both recognized the reference response correctly. These are ASR diagnostics,
not human listening scores. The baseline was then deliberately stopped with epoch 87
saved, and the improved pipeline started preparing its full cache on four GPUs.
The report and WAV files are in `outputs/baseline_switch_check` on the server.
The baseline's unfinished 100-epoch target is cancelled; the v2 target remains 100.

## What the isolation experiment actually found

A snapshot of the baseline at epoch 37 was tested on three held-out conversations.
The results are in `outputs/quality_v2_diagnostics/report.json` on the server.

| Path | ASR word error against reference, three samples | Interpretation |
|---|---|---|
| Original B recordings | 0%, 4.8%, 0% | The recognizer can understand the source recordings. |
| Cached codec latents through frozen decoder | 0%, 0%, 0% | The codec/cache connection is working on these examples. |
| Real B semantic features through trained generator | 0%, 9.5%, 33.3% | The acoustic generator can recover substantial content when given correct semantics. |
| Entire A-to-B generation | ASR returned “I'm sorry.” for all three | The semantic-planning path is the leading bottleneck. |

These are three diagnostic examples, not population-level accuracy measurements.
An appropriate generated response need not match the dataset's exact words. Reference
WER is directly interpretable for reconstruction; for freely generated replies it is
only a diagnostic. Two Whisper sizes are related recognizers, not statistically
independent judges. Human listening remains necessary.

## New representation and data flow

| Component | Input | Output | Purpose |
|---|---|---|---|
| Original SBE and fusion | mel 80D, 3DMM 486D, AU 25D | `[B,Tc,512]`, about 25 Hz | Preserve original audio, appearance and expression branches. |
| Frozen HuBERT on Person A audio | mono waveform at 16 kHz | `[B,Na,768]`, aligned to 50 Hz | Add pretrained speech-content features. |
| Trainable speech adapter | A's 768D features | `[B,Na,512]` | Project, add positions, and contextualize with two transformer layers. |
| Context integration | Original context plus aligned speech features | `[B,Tc,512]` | Keep the downstream 512D interface. |
| Style-conditioned affect transport | Context, A expression, requested style | `[B,Tc,6]` | Give Module 3 access to response style. |
| Semantic planner | Context, affect, style and planned length | `[B,Nb,768]`, 50 Hz | Retain all HuBERT channels instead of a random 256D projection at 12.5 Hz. |
| Acoustic flow model | Predicted or reference semantics plus other conditions | `[B,Tb,128]`, 50 Hz | Produce EnCodec latents. |
| Frozen codec decoder | Codec latents | mono 32 kHz waveform | Convert acoustic features into sound. |

The HuBERT model/revision and per-utterance normalization are pinned to the existing
teacher contract. “50 Hz” means features are interpolated onto an explicit frame grid;
the native convolution boundary has slightly fewer frames. No random feature
projection is used for version 2. Raw B semantics and A speech features are cached
as float16 and loaded as float32. Original mel/3DMM/AU/codec/affect arrays are reused
from the original cache rather than duplicated.

The A waveform and A transcript are resolved from the dataset's conversation ID.
The source JSON's six B responses retain their original style IDs. Splits remain
conversation-based, so the six responses to one A never cross train/val/test.
Cache identity includes source metadata, transcript-file hash, original manifest,
configuration and selected conversations. Target normalization is fitted on train
frames only. Transcripts, waveform samples, feature dimensions, and CTC sequence
lengths are validated; invalid records fail preparation instead of being silently
dropped.

## New training objectives

CTC is a character-sequence loss that learns words without requiring an exact timestamp
for every character. Its vocabulary contains English letters, spaces, apostrophes,
digits, and a blank symbol. Punctuation/case are normalized; digits are retained.
Repeated adjacent characters require an additional blank frame. Impossible alignments
raise an error rather than receiving a zero loss.

| Loss | Applied to | Role |
|---|---|---|
| Input CTC | Trainable A speech adapter → A transcript | Make the adapter retain spoken content. |
| Reference semantic/codec CTC | Real B features → B transcript | Teach the auxiliary heads to recognize content. |
| Predicted semantic CTC | Denoised and sampled B semantics → B transcript | Give the planner an explicit linguistic objective. |
| Predicted codec CTC | Reconstructed clean codec latents → B transcript | Encourage the audio generator to retain planned content. |
| Semantic velocity loss | Noised B speech features | Train a stable diffusion parameterization without dividing by tiny noise-schedule alpha. |
| Codec flow loss | Noised B codec latents | Train acoustic generation. |
| Multi-resolution spectral loss | Differentiably decoded estimate versus real B audio | Supervise acoustic detail at FFT sizes 256, 512 and 1024. |
| Paired relevance contrast | Pooled A context versus B speech semantics | Prefer matching conversation-response pairs; all same-conversation replies are positives. |
| Affect/duration losses | Measured prosody and duration | Retain the original objectives. |
| Neutral affect prior | Valence/arousal/dominance | Discourage extreme values in unlabeled channels; does not create genuine emotional labels. |

Content terms use weight 0.1, spectral loss 0.1, relevance 0.05, and neutral prior 0.01.
Spectral loss uses one valid example every eight training steps to limit GPU cost.
It is omitted from the stochastic validation total; generated validation audio is
evaluated separately. Auxiliary CTC heads are trained heads, not pretrained ASR
teachers. Their losses and gradients have been checked, but that alone does not
demonstrate good speech.

The frozen codec parameters remain frozen. During the differentiable spectral path,
native LSTM kernels are used because cuDNN does not support backward through an
evaluation-mode LSTM. Normal inference continues to use the existing decoder path.

## Training stages and checkpoint separation

The default version-2 schedule is 100 epochs in its own directory:

1. Epochs 1–5: acoustic adaptation with real B semantics and small feature corruption;
   input/reference/acoustic CTC heads, duration, affect and relevance train as well.
2. Epochs 6–10: semantic diffusion and predicted-content supervision; acoustic-generation
   objectives have zero weight during this stage.
3. Epochs 11–100: joint training. Over 20 epochs the probability of replacing a real
   semantic sequence with a generated sequence rises from 5% to 100%.

Generated semantics use four sampling steps during training. The final denoising
step is differentiable, so the downstream codec loss can reach the semantic planner.
Earlier sampling steps are detached to control memory. Evaluation uses 24 semantic
steps and 32 midpoint codec steps. This remaining sampling-budget difference is
explicit; it may need further ablation.

Initialization reuses shape-compatible original SBE, style/voice, duration and parts
of the acoustic generator. The semantic representation, sampler and new heads are
different, so this is a new experiment, not an exact continuation of the v1 optimizer.
Actual v2 resume restores model, optimizer, scheduler, rank-specific RNG, epoch and step.
It requires the same manifest hash, config, and GPU count. Metrics are recorded once
per completed epoch. Stage-specific best checkpoints prevent comparing unlike loss
scales. `best.pt` is still selected by the current stage's validation total, not by an
unvalidated ASR quality threshold.

## Evaluation

Every five epochs, generated validation examples cover six response styles and
different conversations. The report includes raw clipping before output attenuation,
quiet-frame fraction, duration, two ASR transcripts, reference-recording ASR controls,
transcript-reference and transcript-input embedding similarities, and measured pitch,
energy and ASR-derived word rate. The word-rate estimate is unreliable when ASR is
wrong and is labeled accordingly.

A controlled sweep also keeps A, voice group and seed fixed while changing only style.
Swapping A with another conversation tests input sensitivity. Neither test by itself
proves semantic correctness. `human_ratings.csv` provides empty clarity, relevance,
empathy and naturalness rating fields; no human ratings are fabricated.

At the end, 100 unique held-out conversations are evaluated, approximately balanced
across styles, plus the controlled style sweep. Male/female metadata is still not an
individual speaker identity. Valence/arousal/dominance still lack direct annotations.

## Commands on the server

By default, the controller waits for the original baseline to finish. For the current
user-authorized switch, `--accept-stopped-baseline` also accepts a deliberately stopped
baseline whose status confirms that its GPU workers have exited. It then prepares the
separate full cache and runs the staged experiment. Do not start a second controller
if one is already waiting or active.

```bash
.venv/bin/python -u scripts/run_quality_training.py --epochs 100 --batch-size 16 --accept-stopped-baseline
```

Manual data preparation, when no controller is active:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 .venv/bin/torchrun --standalone --nproc_per_node=4 prepare_quality.py --output outputs/bk_quality_prepared
```

Manual training after preparation, when no controller is active:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 .venv/bin/torchrun --standalone --nproc_per_node=4 train_quality.py --manifest outputs/bk_quality_prepared/manifest.json --config outputs/bk_quality_prepared/config.json --initialize-baseline outputs/bk_5epochs/best.pt --output outputs/bk_quality_v2_100 --epochs 100 --batch-size 16 --workers 4
```

Resume by replacing `--initialize-baseline ...` with
`--resume outputs/bk_quality_v2_100/last.pt`; epochs is the total v2 epoch target.

Normal inference requires only A's feature file, A's audio and the requested style/voice:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python infer_quality.py --checkpoint outputs/bk_quality_v2_100/best.pt --features outputs/bk_prepared/sample_000456.npz --audio /home/aisha/bk-dataset/generated_input_audio/hit_45_conv_90_female_Happy.wav --style 0 --speaker 1 --output outputs/quality_new_example
```

`infer_quality.py` does not read B semantic targets, B text, B audio, or ground-truth
response length. Loading a prepared NPZ as its A-feature input does not make those
targets available to generation. Tests explicitly verify this separation.

## Verification completed during implementation

- Three-case codec/acoustic/planner isolation on the real server dataset.
- Five focused tests: CTC alignment, curriculum gradient flow, A-content gradients,
  differentiable codec gradients, inference target exclusion and checkpoint round-trip.
- Existing full-architecture regression tests remain passing.
- Four-GPU, batch-16-per-GPU small-data run through all three stages.
- Four-GPU resume into the next epoch and generated audio/ASR evaluation.

The short smoke experiment is a software check, not evidence of improved speech.
Success is judged from actual generated results after substantive training and human
listening. The richer representation and new losses cost more than the original
three-minute epochs; a full-run ETA must be measured rather than copied from v1.
