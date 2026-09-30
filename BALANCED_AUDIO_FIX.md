# Speech supervision diagnosis and fix

User requested another evidence-based diagnosis and correction on September 29, 2026.

## Completed result

The five-epoch experiment finished all 160 updates on four GPUs. The training loop
took 29.5 minutes; preparation and audio evaluations brought the controller's total
runtime to about 34.6 minutes. All GPUs were released afterward.

| Model | Development WER, 32 cases | Reserved WER, 32 different cases |
|---|---:|---:|
| Preserved acoustic baseline + new vocabulary | 15.65% | 11.99% |
| Corrected supervision, selected epoch 5 | 15.75% | 12.15% |
| Reference recordings | 4.50% | 3.41% |

Development WER by epoch was **15.79%, 17.21%, 16.70%, 16.97%, 15.75%**.
The candidate passed the reserved-set acoustic gate, but failed the development
excess-error threshold and did not improve either set's mean. The fixed overall
decision is therefore **do not replace the baseline; do not start the planner**.

The reserved difference was only +0.16 percentage points. Its descriptive paired-
bootstrap 95% interval was -1.95 to +2.46 points (10,000 resamples, seed 42). This
does not demonstrate a meaningful degradation or improvement on that small set.
The corrected run was more stable than the preceding sparse-supervision run, whose
fifth-epoch development WER was 19.98%, but it did not outperform the original model.

Full-validation frozen-recognizer loss now **decreased** from 0.04021 to 0.03627
(9.8%), while total loss decreased 0.80648 to 0.78782. Independent Whisper WER did
not improve. Thus the coverage correction is verified, but lower evaluator loss
still does not establish clearer speech or relevant responses.

All five candidate checkpoints were finite; only codec-generator weights changed.
The internal content head and every existing frozen component matched initialization,
and the new planner stayed unchanged across epochs. Downloaded source hashes match
local code. Full checkpoints and all audio remain on the server; reports, representative
WAVs and `rejected_candidate_inference.pt` are saved locally in
`outputs/server_training_results/balanced_audio_v1`. The baseline remains preserved.

The broader planner remains untrained. These are reconstruction evaluations with
correct B units supplied, not evidence that a complete A-only response model works.
The bounded experiment is complete; no automatic longer run was started.

Paired confirmation review found seven improved, seven worse and 18 unchanged
transcripts. All generated durations matched and no generated sample clipped or
hit the ASR token limit. Local substitutions persist: one target phrase, “your
senior year felt emptier when your brother left,” was recognized as “you're seeing
your ear felt enter when your burner left” in both models. A smaller error improved
from “really twitching” to “really touching”; another changed “want to connect” to
“walk to connect.” One counted regression was only cancelled/canceled spelling.
These are automatic transcription observations, not direct human listening judgments.

## Concrete problems found

1. The preceding sampled-audio run supervised an actual waveform for only the first
   utterance on each GPU every fourth update. Of 2,048 training records, only **155**
   received this supervision in five epochs. There were 160 waveform exposures versus
   10,240 latent-flow exposures; 120 of 160 updates had no waveform objective.
2. Frozen-recognizer validation loss increased from 0.04678 to 0.06249 (33.6%) while
   latent-flow loss decreased. The total loss concealed the content regression.
3. Waveform validation also evaluated only the first utterance in each rank's batch.
   The independent 32-case Whisper evaluations were complete, but this sparse training
   diagnostic could not describe all validation examples.

These are observed coverage and measurement defects. They are plausible contributors
to the regression, not proof that fixing them solves response generation.

## Gradient evidence

Sixteen representative training batches were evaluated without updating weights.
Mean weighted gradient norms were 0.360 for flow, 4.743 for sampled CTC, and 1.012 for
sampled spectral loss. Flow alone opposed the CTC direction in 6/16 cases; the combined
gradient favored CTC in all 16. Therefore it would be inaccurate to claim the CTC
gradient was too small whenever present. The evidence supports removing flow-only
updates and increasing the number of utterances, while keeping the loss weights fixed.
These are local gradient directions, not a guarantee about Adam updates or unseen data.

## Implemented correction

- Sampled waveform objectives run **every update**, on four distinct utterances per
  GPU. Indices rotate within each shuffled batch. Global batch stays 64 (16/GPU).
- The deterministic five-epoch schedule provides 2,560 waveform exposures covering
  1,566 unique training records (76.46%), versus 155 previously. This is increased
  coverage, not supervision of every training record.
- Validation computes the waveform objectives on **every** validation utterance,
  with the correct transcript, waveform length and conditioning for that utterance.
- Keep flow weight 1, CTC weight 0.05 and spectral weight 0.1, learning rate 3e-6,
  original acoustic initialization and the same 1,024-unit training-only vocabulary.
- Continue differentiating all 32 sampler steps through the frozen codec and ASR.
  Only the codec generator is trainable. Module 3 and all other components stay fixed.
- Explicit flow weight and coverage are stored in version-2 checkpoint recipe metadata;
  incompatible earlier runs cannot silently resume under the new objective.
- Per-utterance selection trims every temporal input to its valid length. A new
  mixed-length regression test caught and verified this necessary padding correction.

Twenty-six focused tests passed locally and on the server. Four-GPU forward/backward
preflight passed, gradient norm 1.55936, peak allocated memory 6.81 GB/GPU.

## Data and architecture checks

No sampler sign error, detached full-sampler gradient, codec-normalization mismatch,
or corrupted response ownership was found. Earlier auditing verified all 2,176
cached response waveforms against source recordings.

An additional A-input duration audit found median audio/video relative duration
spread 0.40%, 95th percentile 1.35%; one extremely short example exceeded 5%, and none
exceeded 10%. This does not support a widespread input timing mismatch as the main
cause. It does not prove phonetic or facial-event alignment.

There is a separate **response-planning limitation**: the older 64-example unit
planner reached 100% training unit accuracy but 2.80% validation accuracy. Swapping
A's context changes predictions, so A is used; the planner memorized its training
set. The broader acoustic checkpoints contain a freshly initialized, frozen planner
because broader planner training has not yet passed its preceding acoustic gate.
They must not be presented as trained end-to-end A-only response models. Improving
reconstruction alone cannot establish relevant or empathetic replies.

## Prespecified trial

Five epochs / 160 updates on 2,048 training records, four GPUs, 16 examples/GPU.
Save all five candidates and compare actual speech on the same 32 development cases.
Then evaluate the selected candidate and baseline on the last 32 previously unused
confirmation recordings within the selected 128 validation records. Both earlier
confirmation groups are excluded. These records had old aggregate validation exposure,
so this is not a pristine final test. Original test audio remains unused.

The unchanged gate requires reference mean WER <=20%, generated mean <=20%, excess
over references <=10 percentage points, and >=75% of examples within 20 points of
their reference. Promote only if the candidate passes both sets and is no worse
than baseline on both. Do not change the threshold after seeing results.

Server outputs: `/home/aisha/bk-full-architecture/outputs/balanced_audio_v1`.
Controller: `scripts/run_acoustic_controls.py --balanced-experiment`.
Earlier checkpoints remain preserved. Evaluation uses independent Whisper ASR;
human naturalness, pronunciation, empathy and response relevance are not established
by reconstruction WER.
