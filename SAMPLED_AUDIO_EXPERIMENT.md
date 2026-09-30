# Training against actual sampled audio

User-authorized experiment, September 29, 2026. This follows the failed low-learning-
rate/teacher ablations documented in `ACOUSTIC_CONTROLLED_CHECK.md`.

## Completed training and development results

All five epochs / 160 updates completed. The optimizer loop took approximately
145 seconds, excluding data loading, initial validation, and audio evaluations.

| Checkpoint | Development reconstruction WER |
|---|---:|
| Preserved baseline + new vocabulary | 15.65% |
| Sampled-audio epoch 1 | 16.31% |
| Sampled-audio epoch 2 | 18.75% |
| Sampled-audio epoch 3 | 18.78% |
| Sampled-audio epoch 4 | 19.83% |
| Sampled-audio epoch 5 | 19.98% |

The first epoch was the best trained candidate. None passed the development gate
or improved on the preserved baseline. The first epoch was better than the earlier
teacher-on adaptation's 17.95%, but that is not evidence of improvement over the
original model. Several aspects of the recipe changed together, so this experiment
does not isolate the effect of the sampler from freezing the content head or dropout.

All five candidate checkpoints had finite tensors. Only generator parameters
changed: the existing context, Module 3, content head and other frozen tensors
matched the initializer; the new planner stayed identical across epochs. The
strict generic checkpoint loader successfully produced the evaluation audio.
Local and server source hashes matched for the modified training and model files.

## Reserved confirmation and final decision

| Model | Development WER | New confirmation WER |
|---|---:|---:|
| Preserved baseline | 15.65% | 11.57% |
| Sampled-audio epoch 1 | 16.31% | 13.13% |
| Reference recording | 4.50% | 1.37% |

The candidate failed both gates and did not improve either mean. The confirmation
increase was +1.56 percentage points; a descriptive paired-bootstrap 95% interval
was +0.009 to +3.62 points (10,000 resamples, seed 42). This small, single-seed
experiment is insufficient for broad claims about sampled-audio training generally.
The original baseline also narrowly missed the confirmation excess-error gate:
10.20 points over the references versus the allowed 10. The gate was not changed.

**Decision: reject this adaptation and retain the original acoustic checkpoint.**
The new training path is implemented and its gradients verified, but this weighting,
sparse waveform coverage and five-epoch recipe did not improve speech. Do not equate
successful backpropagation or completed epochs with successful audio generation.
No planner or full 100-epoch run was started. All four GPUs were released.

Full checkpoints and all evaluation WAVs remain on the server. Reports, selected
representative WAVs and an explicitly named `rejected_candidate_inference.pt` are
downloaded under `outputs/server_training_results/sampled_audio_v1`. The original
checkpoint and vocabulary remain preserved at their existing locations.

Any follow-up should first measure how the flow-anchor gradient competes with the
sampled speech objectives and test waveform-supervision coverage, rather than
assuming that longer training will help. That follow-up is not part of this completed
experiment. Human listening is still needed to assess naturalness and pronunciation.

## What changed

Previously, waveform losses examined a one-step estimate formed using ground-truth
codec latents mixed with noise. This experiment generates from Gaussian noise through
the **same 32-step midpoint sampler used at inference**. All 64 velocity-network calls
remain differentiable. Activation checkpointing recomputes intermediate activations
during backward to reduce memory; it does not detach earlier sampling steps.

Generated latents pass through the frozen EnCodec decoder and then a frozen pretrained
wav2vec2 speech recognizer. Its CTC loss compares recognized content with B's training
transcript. Spectral loss compares the sampled waveform with B's reference audio.
The existing flow-matching objective remains an anchor on all 64 examples per update.

| Component | Updated? | Role |
|---|---|---|
| Codec latent generator, Module 6 | Yes | Generate understandable codec latents |
| EnCodec decoder, Module 7 | No | Decode latents; transmit gradients to generator |
| External wav2vec2 content evaluator | No | Score generated words; transmit gradients |
| Internal codec content head | No | Excluded from this experiment's optimized losses |
| Context encoder, Module 3, semantic planner, duration predictor | No | Preserve existing conditioning |

The generator uses evaluation-mode dropout even during optimization, matching
inference behavior while keeping its weights trainable. This flag is separate from
gradient tracking. The final waveform is decoded per valid utterance, avoiding padded
audio boundaries. Native recurrent kernels enable differentiation through the frozen
codec; inference may use cuDNN and therefore has small numerical differences.

The new loss is:

`flow_matching + 0.05 * sampled_waveform_CTC + 0.1 * sampled_waveform_spectral`

Waveform objectives use one utterance per GPU every four updates. Flow matching uses
all examples. Validation now includes sampled waveform objectives on the first
utterance of each rank's validation batch; this is a sparse loss diagnostic, **not**
the all-example quality selection score. Independent Whisper ASR evaluates all 32
development recordings for checkpoint selection.

B units, audio and transcript are deliberately supplied for acoustic supervision.
This tests whether the generator can speak specified content. It does not train the
new response planner or demonstrate an appropriate reply generated from A alone.
Normal inference still uses the A-only input whitelist and no B transcript.

## Checks before training

- Local and server tests: 21 passed, covering earlier model invariants, waveform agreement,
  sampled-loss gradients, frozen evaluators and validation waveform terms.
- Real pretrained-model test: nonzero finite sampled-CTC gradients reached only the
  generator; no gradients reached the codec, recognizer or other model components.
- The real training waveform and ordinary inference output differed by maximum
  absolute amplitude 0.000626 and relative RMS 0.0452%, within the explicit bounds
  of 0.001 absolute amplitude and 0.1% relative RMS.
- Four-GPU forward/backward check passed on all ranks with batch 16 per rank.
  Gradient norm was 1.44994; peak allocated memory was at most 3.12 GB per GPU.
- A preflight exposed mixed-precision resampling feeding reduced-precision audio to
  the float32 recognizer. The entire recognizer preprocessing path now stays float32,
  and a regression test covers that case. The failed preflight made no weight updates.
- The prior audit of all 2,176 selected train/validation records passed waveform
  identity, response ownership and frame-length checks; the same data is reused.

## Fixed experiment protocol

- Initialize from `outputs/quality_recovery/acoustic/best.pt`, not rejected adaptations.
- Reuse the 1,024-unit vocabulary fitted on 2,048 training conversations.
- Four GPUs, batch 16 per GPU, global batch 64, five epochs = 160 optimizer updates.
- Learning rate 3e-6, existing warmup/cosine schedule, gradient norm clipping at 1.
- Same 32 development recordings; save all five epoch candidates and select by
  actual generated-audio WER, not the training or validation objective.
- Reserve another 32 validation recordings, excluding the 32 confirmation recordings
  already used by the previous experiment. Evaluate only baseline and selected
  candidate after selection. These records contributed to older aggregate validation
  losses, so they are not a pristine final test. Original test audio remains unused.
- Retain the original checkpoint unless the candidate passes both fixed gates and
  improves or matches baseline WER on both sets. Do not retune using confirmation.
- The acoustic gate requires reference mean WER <=20%, generated mean <=20%, excess
  <=10 percentage points, and >=75% of cases within 20 points of their reference.

This is a change of training objective and freezing policy, not a guaranteed quality
improvement. ASR checks do not establish naturalness, empathy or response relevance.

## Running and outputs

From `/home/aisha/bk-full-architecture`, after the real gradient preflight:

```bash
.venv/bin/python -u scripts/run_acoustic_controls.py --sampled-experiment
```

Do not launch a duplicate. The controller uses the shared experiment lock and resumes
compatible checkpoints. Outputs are under `outputs/sampled_audio_v1`, including
`real_gradient_check.json`, `preflight`, `protocol.json`, `sampled/step_*.pt`, five
audio-evaluation directories, confirmation reports and `decision.json`.
