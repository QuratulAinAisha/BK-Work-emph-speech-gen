# Recovery experiment results — September 29, 2026

Follow-up: [GENERALIZATION_CHECK.md](GENERALIZATION_CHECK.md) records the expanded
2,048/128 data audit and 32-example acoustic checks. Those broader controls did not
pass the existing gate; the measurements below remain the earlier small-pilot results.

The staged recovery and conditional Module 5 experiment are complete. The audio
generator improved, and the discrete planner can reproduce learned A-to-B responses.
**Reliable responses to unseen inputs have not been demonstrated.** Full-data joint
training has not been restarted; the old runs and all pilot checkpoints are preserved.

## Measured results

All percentages below are mean reference word error rates from Whisper base.en.
Lower is better. Eight examples were used for each audio check. WER can exceed 100%
because of inserted words; it is not an accuracy percentage or a human listening score.

| Controlled check | Result | Conclusion |
|---|---:|---|
| Real validation B recordings | 2.69% | The reference control is mostly recognizable. |
| Codec reconstruction on the same validation set | 1.25% | Cached codec features and the fixed decoder preserve the words. |
| Correct continuous B features, before acoustic recovery | 14.06% | Initial matched acoustic control. |
| Correct continuous B features, after acoustic recovery | **8.91%** | Acoustic recovery passed. |
| Continuous planner, A-only, eight training cases | 206.16% | Could not reproduce learned responses; irrelevant/repetitive transcripts. |
| 512 correct B units, before / after acoustic adaptation | 17.17% / 33.04% | Rejected this adaptation; did not train its planner. |
| 1,024 correct B units, frozen improved acoustics | **11.41%** | Passed the unit-to-audio interface gate. |
| Discrete planner at step 1,200, A-only, eight training cases | **12.32%** | Passed the small-set reproduction gate. |
| Discrete planner at step 100, A-only, eight validation cases | 95.23% | Validation-selected checkpoint still gave poor new responses. |
| Discrete planner at step 1,200, A-only, the same validation cases | 94.43% | Extra memorization did not solve new-input response generation. |

Oracle checks deliberately supply the correct B speech features. They test the audio
path, not the ability to choose a response. The A-only checks supply only A's features,
requested style and voice group; the response duration is also predicted.

The unit prediction diagnostic covered **all 64 training and 24 validation examples**.
With correct B length supplied solely to isolate unit prediction, the final planner
matched 100% of training unit frames and only 2.80% of validation frames. The most
common-unit validation baseline was 0.56%. Swapping A's context changed 97.85% of
training predictions and 94.07% of validation predictions. Thus the planner uses A,
but input dependence alone does not establish understanding or a correct reply.

## Concrete examples

**Learned training example, generated from A only:**

- A: “So the other day my daughter went potty for the first time on her own!”
- ASR transcription of generated B: “That's wonderful. You must feel so proud seeing
  her take that big first step all on her own.”
- The recognized words match this training target. This demonstrates learned
  reproduction, not performance on an unseen conversation.

**Held-out validation example, same final checkpoint:**

- A: “I had been doing poorly in college for a while.”
- ASR transcription of generated B: “It sounds good.”
- That response does not appropriately acknowledge A's difficulty. The failure is
  evident from the content as well as the reference-based metric.

The final checkpoint also produced short generic phrases such as “That sounds good”
and “I'm sorry” for other held-out inputs. No human naturalness or empathy ratings
are claimed. ASR can mishear or hallucinate; listening to the saved WAVs remains useful.

The normal `infer_quality.py` command was also run on the actual A waveform plus its
mel/DMM/AU features for the two examples above. It reproduced the same respective
transcripts, without reading cached B features or a reference duration. These files
are under `units_1024/raw_a_inference` in the collected results.

## What changed and what was verified

- Fixed validation conditions, matched semantic sampling budgets, explicit frozen
  stages, warmup/cosine learning rates, and full 0–1 codec flow times.
- Added a strictly loaded, pinned, frozen waveform ASR teacher and checked its
  reference recognition and gradients before using it in acoustic adaptation.
- Tested the original continuous planner before replacing it. Its failed reproduction
  check triggered the discrete-unit experiment.
- Fitted codebooks to training features only. The accepted interface has 1,024 units,
  768-dimensional center vectors and 50-Hz frames. The final test split was untouched.
- Used masked parallel unit prediction in Module 5, preserving the existing SBE
  dimensions, Module 3's six controls, acoustic generator and fixed codec.
- Kept every non-planner/non-duration component frozen during unit-planner training.
- Passed 17 focused tests locally and on the server, including gradient flow, frozen
  weights, checkpoint loading, A-only inference and unit prediction contracts.
- Passed actual four-GPU save/resume checks. Training used batch 16 per GPU, global 64.
- Audited both unit checkpoints: weights are finite, and all changes during planner
  training are confined to the planner and duration predictor (77 changed tensors).
- Added short English and Korean comments to the implementation.

The pilot used one style and one unverified voice group. It does not establish the
behavior of every style or all six affect controls. Valence, arousal and dominance
still lack direct labels in this dataset.

## Interpretation and next training decision

The small experiment has answered the architectural question: the discrete planner
can learn the paired response sequences that the continuous planner failed to learn
under its tested recipe. It has not answered the generalization question. Training
on 64 conversations is a memorization diagnostic, not a substitute for broad dialogue
training, and its train/validation gap is substantial.

The next useful experiment is broader dialogue training of this discrete planner,
with acoustics frozen initially and scheduled A-only audio evaluation. Before applying
it to all styles and voices, repeat the unit-oracle controls on those groups. Fit any
replacement codebook on training data only and treat a changed codebook as a new
representation. Joint acoustic updates should remain gated by actual reconstruction
and response checks. There is no evidence here that simply completing another 100
epochs would guarantee an appropriate, empathetic response.

## Where the work is saved

Server project: `/home/aisha/bk-full-architecture`.

- Complete recipe and commands: `README_quality_recovery.md`.
- Continuous recovery: `outputs/quality_recovery/acoustic` and `planner`.
- Rejected 512-unit trial: `outputs/quality_recovery/units`.
- Accepted interface and tested unit planner: `outputs/quality_recovery/units_1024`.
- Local reports, selected inference checkpoints and WAV examples:
  `outputs/server_training_results/quality_recovery`.

The latest unit checkpoint is the training-reproduction checkpoint, not a validated
general-purpose response model. Its validation-selected counterpart is also saved.
Local checkpoint exports omit optimizer/RNG state; use the full server checkpoints
for training resume. Local export hashes were verified. The bounded experiments and
final inference checks have finished; the GPUs are released.
