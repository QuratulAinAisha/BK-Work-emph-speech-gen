# Controlled acoustic checks

User-approved follow-up to `BROAD_UNITS_RUN.md`, September 29, 2026.

## Development results

Both five-epoch trials completed on four GPUs with batch 16 per GPU. All ten
checkpoints were evaluated using generated audio, and none beat the preserved
baseline (15.65% WER) or passed the development gate.

| Epoch | Teacher off, WER | Teacher on, WER |
|---|---:|---:|
| 1 | 18.66% | 17.95% |
| 2 | 20.49% | 18.32% |
| 3 | 22.15% | 20.34% |
| 4 | 21.75% | 20.36% |
| 5 | 22.11% | 20.59% |

The least-bad trained candidate is teacher-on epoch 1. It is a candidate for the
prespecified confirmation comparison, **not an approved replacement**. Removing
the waveform teacher does not fix this regression. Keeping it performed better
in this single-seed experiment, while both variants still degraded from the
baseline. Lowering the learning rate tenfold also did not solve the problem.

The two runs had identical initial validation values. Every saved model tensor
was finite. All non-acoustic tensors, including the newly initialized planner,
were unchanged across the ten candidates; existing frozen components also
matched the original initialization. All 17 focused tests passed locally and
on the server. Source file hashes matched between local and deployed scripts.

**Planner decision:** do not start the broader planner, because the development
acoustic gate failed. Preserve the earlier acoustic checkpoint and new codebook.

## Reserved confirmation results

| Path | Development WER (32) | Reserved confirmation WER (32) |
|---|---:|---:|
| Preserved acoustic baseline + new vocabulary | 15.65% | 11.13% |
| Selected trained candidate: teacher-on epoch 1 | 17.95% | 11.58% |
| Reference audio | 4.50% | 2.25% |

Both models passed the acoustic gate on the confirmation subset, but the trained
candidate failed on development and did not improve either subset's mean. The
prespecified overall decision therefore remains **reject adaptation**. Do not
change the gate or choose another checkpoint using confirmation results.

The confirmation difference was only +0.45 percentage points; its descriptive
paired-bootstrap 95% interval was -1.18 to +2.19 points (10,000 resamples, seed 42).
This is not convincing evidence of a confirmation-set degradation or improvement.
Teacher-on versus teacher-off differences also have substantial uncertainty on
this small sample and single training seed. There is no basis to claim that the
external waveform teacher alone caused the earlier failure.

All trial processes completed and the four GPUs were released. No broader planner
or 100-epoch training was started. Full checkpoints and all generated controls
remain on the server. Reports, selected representative audio and a clearly named
`rejected_candidate_inference.pt` export are downloaded under
`outputs/server_training_results/acoustic_controlled_v1`.

## What the next experiment should change

The tested data-matching checks passed, and simply removing the teacher or reducing
the learning rate did not fix the regression. The next proposed experiment should
supervise the actual sampled acoustic output, with a frozen content evaluator, and
retain audio-based checkpoint selection. Currently the codec content head also
learns alongside the generator, so reducing that proxy loss need not demonstrate
better intelligibility. These are hypotheses to test, not fixes already shown to
work. Human listening is still needed for pronunciation, distortion and naturalness;
reconstruction WER cannot establish relevance or empathy of A-only responses.

## What was damaged in the previous adaptation?

On the same 32 examples, 19 had higher word error, three improved, and ten were
unchanged. ASR alignment counts changed as follows:

| Error | Before | After |
|---|---:|---:|
| Substituted words | 53 | 91 |
| Deleted words | 11 | 16 |
| Inserted words | 18 | 22 |

Output durations were identical. No ASR generation hit its token limit. Maximum
raw clipping fraction after adaptation was 0.0000209 (0.00209%). These checks do
not support duration changes or substantial clipping as the principal explanation.
They do not replace human listening or establish precise pronunciation errors.

For example, the correct phrase “it makes sense you're upset” was recognized
correctly before adaptation, but as “at mixed interrupts” afterward. Another
“that build-up must feel great” became “that the final death must be alright” in
ASR output. These are reconstruction diagnostics with correct B units supplied,
not outputs from a trained new response planner.

## Objective mismatch found in code

`quality.py` computes waveform spectral/teacher losses only during training, on
the first sample per rank every four steps. Validation excludes these waveform
losses. The teacher supervises a one-step clean-latent estimate, whereas inference
uses the generator's sampling procedure. Therefore validation loss is not a direct
measurement of full generated audio. This is an observed implementation difference;
whether the teacher caused the regression requires the matched comparison below.

## Alignment audit

`scripts/audit_acoustic_alignment.py` checks all 2,176 selected records against the
source metadata, response-audio owner, actual waveform resampled to 32 kHz, and
codec/semantic frame counts. It verifies the normal frame-rounding tolerance.
It does not prove phonetic alignment. Preparation linearly interpolates HuBERT's
natural feature length to the codec frame count, preserving all 768 channels.
The controller stops on an audit failure.

**Result:** all 2,176 examples passed. Maximum cached/source waveform difference
was exactly zero; maximum frame-duration rounding was 18.65625 ms. Source ownership,
styles, voice-group IDs and response text matched. These findings rule out the
checked metadata, waveform and length mismatches, not all phonetic timing errors.

## Prespecified comparison

- Preserve the new 1,024-center codebook, trained on 2,048 training conversations.
- Initialize both trials from the preserved pre-adaptation acoustic checkpoint.
- Same training records, seed, batch order, optimizer schedule, four GPUs, batch
  16 per GPU, five epochs / 160 optimizer updates, learning rate **3e-6**.
- Trial A: waveform ASR teacher weight **0**. Trial B: weight **0.05**.
- Keep spectral and latent/content losses unchanged. The comparison removes only
  the external waveform ASR teacher; it does not remove every content/CTC loss.
- Freeze Module 3, context, new planner and the external codec. Train only the
  codec generator and codec content head.
- Save every epoch and evaluate all ten candidate checkpoints on the same 32
  development examples. Select by actual oracle reconstruction WER, not loss.
- Reserve another 32 validation recordings before running trials. Evaluate only
  the chosen candidate and original baseline there, after selection.
- These confirmation recordings have not had audio quality evaluated previously,
  but participated in older aggregate validation losses. They are not an untouched
  final test. Original test audio remains unused.

The winning candidate must pass both development and confirmation gates and be
no worse than the original baseline on either. The unchanged gate requires mean
oracle WER <=20%, excess over reference <=10 percentage points, and at least 75%
of cases within 20 points of their reference. Reference WER must also be <=20%.

Only a confirmed pass makes the broader planner eligible for its bounded five-epoch
trial with frozen acoustics and A-only response checks. A failed acoustic experiment
must not trigger another automatic long training run. Passing ASR checks does not
establish empathy, naturalness, appropriate response meaning, or generalization.

## Artifacts

Server: `/home/aisha/bk-full-architecture/outputs/acoustic_controlled_v1`.

- `protocol.json`, `alignment_audit.json`, `run_status.json`.
- `no_teacher` and `teacher`: training checkpoints, recipes and loss curves.
- `*_epoch_1` through `*_epoch_5`: actual reconstruction reports and WAVs.
- `selected_candidate.json`, `confirmation_*`, `decision.json`.

The previous comparison is saved locally at
`outputs/server_training_results/broad_units_v1/paired_error_analysis.json`.
Existing 17 focused model/recovery tests passed after adding candidate retention;
the model's loss and inference implementations have not been changed in these trials.
