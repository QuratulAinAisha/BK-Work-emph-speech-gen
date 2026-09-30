# Architecture problems, corrections, and experiment progress

**Snapshot date: 30 September 2026.** This report consolidates the development history into one current record. It describes what was actually implemented and measured, including unsuccessful experiments. It does not treat proposed work as completed work.

The code executes the seven-stage speech pipeline. Reconstruction and small-set learning improved. **Reliable empathetic responses to unseen Person-A inputs remain unsolved.** No candidate from the final planner investigation was promoted. The prepared 40-clip blinded listening package has not received human ratings. Final-test audio/features were not evaluated in that investigation.

## 1. What was present in the uploaded code

The starting archive was `2026BK-team1-integrated-baseline.zip`, SHA-256 `09c9498e82706fed889cdbbbd1fffe1d86843f8443ae9df15666b9c7497ed211`.

It already contained the integrated speaker-behavior encoder, three-modality fusion, a conditional mel decoder, conversation-based data splitting, and shape/padding tests. It was not an implementation of the complete new codec-based architecture diagram.

For this release, the original archive was compared directly with the workspace. `model/speaker_behavior_encoder.py`, `dataset/empathy_dataset.py`, `model/conditional_mel_decoder.py`, `tests/test_sbe_contract.py`, and `tests/test_split_manifest.py` remain byte-for-byte unchanged. Therefore the original conversation split, SBE padding handling, and the mel decoder's existing positional-encoding correction are **inherited safeguards**, not fixes newly invented in this project.

| Starting limitation or mismatch | Check / decision | Implemented change |
|---|---|---|
| No explicit affective response transport stage | Preserve the original SBE output contract | Added a temporal Transformer producing six response-affect controls from A context/expression; later added requested style conditioning |
| No separate response-duration planner | Response lengths must be predicted from available inputs | Added masked pooling and an MLP predicting bounded log-duration; derive unit/codec counts from rates |
| Original output path predicted mel features rather than the diagram's codec latents | Preserve useful encoding, implement the new output flow | Added semantic planning, conditional acoustic flow generation and a frozen EnCodec decoder |
| Diagram says 58D appearance; repository uses 486D | Inspect actual projection weights and existing tests | Retained 486D by default; separate diagram-input configuration requires compatible features/weights |
| No trained SBE checkpoint was supplied | Avoid labeling random initialization as pretrained | Implemented explicit train-from-scratch initialization; later research stages froze the trained initializer |
| Six-channel B emotion trajectories were undocumented | Use only defensible targets | Derived pitch, energy and speaking rate; retained missing-label masks for unsupported emotional channels |
| Response audio identities were undocumented | Gender/face identity cannot certify speaker identity | Used existing voice-group IDs with this limitation stated |
| Shape-correct output could still be meaningless speech | Test actual generated audio and reference controls | Added independent ASR, oracle controls, context perturbations, waveform checks and blinded listening preparation |

These are architectural gaps and data limitations. They are not all programming bugs in the supplied baseline.

## 2. How the implemented system changed

| Component | Current implementation | Important boundary |
|---|---|---|
| Modules 1–2 | Original SBE/fusion, plus A-side frozen HuBERT features and a trainable speech adapter | Fusion/context stays 512D; original appearance width stays 486D |
| Module 3 | Lightweight temporal affect Transformer | At inference it sees A and requested style; B measurements supervise training only |
| Module 4 | Masked mean pooling and duration MLP | Ordinary inference predicts duration; true B duration is a labeled diagnostic control |
| Module 5 | Training-fitted discrete speech units with masked refinement; separate causal feasibility branch | 1,024 units, 768D centers, 50 Hz; these are learned speech clusters, not verified word/phoneme symbols |
| Module 6 | Conditional flow/DiT acoustic generator | Produces 128D codec latents at 50 Hz, normally using 32 sampling steps |
| Module 7 | Pinned pretrained EnCodec decoder | Produces mono 32 kHz audio; decoder weights are fixed |
| Training/evaluation | Staged objectives, strict loading, frozen-module audits, reproducible sampling and selections | A-only generation is kept separate from B-assisted reconstruction |

The first complete implementation used 256D projected continuous semantic targets at 12.5 Hz. Later content-supervised work retained native-width 768D HuBERT features at 50 Hz, then introduced the discrete unit interface. These configurations remain identifiable for checkpoint compatibility. They are implementations within one codebase, not separate packaged releases.

The six controls are valence, arousal, pitch, energy, speaking rate and dominance. Pitch and energy come from B audio; speaking rate uses B's transcript word count and duration. Valence, arousal and dominance lack verified direct targets. A neutral regularizer is not a substitute for emotion labels. Varying these controls is not proof that all six have learned their intended meanings.

## 3. Dataset construction and use

| Item | Observed setup |
|---|---|
| Original corpus | 10,000 A conversations, six styled responses each, 60,000 expanded rows |
| Original split | 8,500 train / 1,000 development / 500 test conversations |
| Grouping | Split by conversation before expanding styled replies |
| A inputs | Mel 80D, appearance 486D, expression 25D, and A waveform-derived HuBERT 768D |
| B training targets | Response waveform, HuBERT features/unit IDs, codec latents, duration, available prosody and transcript |
| Small recovery selection | 64 train / 24 development conversations |
| Main controlled selection | 2,048 train / 128 development conversations |
| Expanded B prior | 3,072 train conversations; 3,135 were eligible under the restricted selection rules |
| Controlled scope | Style 0, voice group 1, response duration 3–8 seconds; numeric B text excluded where the frozen teacher alphabet required it |
| Codebook/normalization | Fit on training features only |
| Ordinary inference | A features/waveform plus requested style and voice group; no B transcript, units, waveform or true duration |

The final 32-case confirmation selection excluded 228 conversations already exposed in recorded detailed development checks. Selection scanned 1,285 saved JSON/JSONL reports and applied lexical near-duplicate filtering against all 8,500 training A transcripts and previously exposed development inputs. Eight cases were used for matched audio. This was fresh detailed development evaluation, not a pristine final test: earlier aggregate validation had already used the development split.

Data limitations remain: generated replies are highly templated, some B replies infer emotions not explicit in A text, and voice identity is uncertain. A text-only audit cannot resolve every audio/video emotion judgment. The top three two-word response prefixes cover about 62% of the controlled training replies; "that sounds" alone covers about 35.84%.

## 4. Confirmed implementation/training issues and corrections

| Issue found | How it was checked | Change made / outcome |
|---|---|---|
| Early training/validation conditions changed across curriculum stages | Compare which semantics, sampler budgets and flow times were used | Explicit recovery stages, fixed within-stage validation, matched planner sampling, warmup/cosine scheduling and full 0–1 acoustic flow-time coverage |
| Lower aggregate loss concealed poorer speech | Compare reference, codec, oracle-semantics and A-only waveforms | Select/reject using matched audio controls, not total loss alone |
| Internal content heads could adapt alongside poor generated features | Frozen external recognizer controls | Added a frozen, strictly loaded wav2vec2 waveform evaluator; independent Whisper evaluation remained separate |
| Teacher checkpoint used incompatible weight-normalization names in the installed stack | Strict checkpoint loading and real-reference/gradient checks | Translated known legacy names rather than silently accepting reinitialized teacher weights |
| One-step/estimated-clean objectives differed from actual sampled inference | Compare decoded estimates with full sampler output | Differentiated actual multi-step generated waveforms through frozen acoustics/codec/evaluator in explicit experiments |
| Sampled waveform supervision initially reached too few examples | Count per-update and per-record exposure | Acoustic correction used four rotating examples per GPU every update; coverage rose from 155 to 1,566 of 2,048 records over five epochs |
| Sparse waveform validation did not cover all examples | Inspect validation selection | Acoustic corrected validation evaluated every selected utterance |
| Mixed-length waveform selections retained inappropriate padded extents | Regression test on mixed lengths | Trim each temporal input to its selected utterance's valid length |
| Gradient-enabled attention differed numerically from the ordinary inference path | Same-weight, same-noise waveform parity calibration | Use a cancellation-safe straight-through expression and measured tolerances; calibrated identical-kernel comparison was exact |
| Discrete unit selection blocks ordinary gradients | Verify actual planner gradients and frozen boundaries | Explicitly biased final-step straight-through estimator for the sampled planner experiment; limitation documented |
| The two-example regeneration helper supplied a string where a torch device object was required | First regeneration failed before successful output | Converted to `torch.device`; rerun succeeded with identical generated WAV hashes |

Increasing waveform coverage and fixing the measured issues did **not** establish a response-quality improvement. Successful gradients and correct code execution are separate from learning the desired behavior.

## 5. Experiment progression before the final planner investigation

WER below is reference word error rate from automatic speech recognition. Lower is better for reconstructing a specified utterance. Different valid free replies may have high reference WER, and insertions can push it above 100%. Each row is a within-experiment comparison; different selections must not be combined as one learning curve.

| Stage | What was run / checked | Observed result and decision |
|---|---|---|
| Seven-stage construction | Synthetic shape/gradient checks, real pretrained codec inference, GPU preparation and smoke training | Established connected execution, masks, losses and resume support; synthetic output did not demonstrate learned speech |
| Initial long baseline | The requested training target was extended to 100 epochs; latest audio checked during the run | An epoch-86 snapshot produced almost no useful content; deliberately stopped with epoch 87 saved |
| Content-supervised extension | Native 768D HuBERT targets, A speech adapter, input/output CTC, relevance and spectral objectives | The next long run was stopped with epoch 32 saved; A-only responses remained poor |
| Component isolation | Six styles; real B, codec reconstruction, correct B semantics and A-only generation | Mean WER about 6.7%, 8.6%, 31.3%, and around/above 100%, respectively; content planning was the leading unresolved path |
| Small acoustic recovery | 64/24 selection; 400 acoustic updates; fixed external speech teacher | Matched eight-case correct-semantics WER improved 14.06% → 8.91%; acoustics then frozen |
| Small continuous planner | 600 updates; train and held-out A-only checks | Train WER 206.16%, development 167.49%; failed even the reproduction gate |
| Discrete interface controls | 512-unit adaptation versus 1,024-unit lookup through unchanged improved acoustics | 512-unit oracle WER worsened 17.17% → 33.04%; rejected. 1,024 units reached 11.41% and passed the bounded development interface gate |
| Small masked-unit planner | 1,200 updates on 64 training conversations | Training A-only WER 12.32%; held-out 94.43% at the endpoint. Learned reproduction, not generalization |
| Broader readiness | 2,048/128 selection; 32 matched acoustic examples | Reference 4.50%, codec 5.80%, continuous oracle 14.88%, old-codebook oracle 18.98%; broader gate not passed |
| Broader codebook/acoustic run | Refit 1,024 centers on the broader training selection; five acoustic epochs | New-unit initial WER 15.65%; after adaptation 24.60% despite validation loss 0.97038 → 0.89081. Rejected adaptation |
| Matched acoustic teacher controls | Teacher off/on, five epochs each; reserved confirmation | Teacher-on epoch 1 gave 17.95% development WER versus baseline 15.65%; confirmation 11.58% versus 11.13%. No confirmed gain |
| Full sampled acoustic audio | Five epochs / 160 updates, frozen external content evaluator | Epoch-1 development WER 16.31%, epoch-5 19.98%; reserved 13.13% versus baseline 11.57%. Rejected |
| Coverage/padding correction | Four rotating waveform examples per GPU every update, full waveform validation | Development 15.75% versus 15.65%; reserved 12.15% versus 11.99%. Evaluator loss improved, independent speech error did not |
| B-only prior then conditional planner | Reconstruction pretraining followed by paired response learning | No reliable held-out response gain; prior-assisted examples still gave inappropriate positive/generic phrases |
| A-data/content probes | 128 A recordings and several trained readers of A feature representations | Source-recording ASR WER 4.11%; no evidence of widespread A/transcript mismatching. Recovering words did not establish dialogue understanding |
| Original-rate A memory | Fused, downsampled/restored speech, and native 50Hz speech memory; 320 matched updates each | Native-minus-resampled development unit accuracy −0.0265 percentage points, confirmation +0.0478; intervals included zero. No confirmed useful gain |

The repeated acoustic failures motivated preserving the stronger acoustic initializer and investigating how Module 5 learns B structure and A-conditioned response content.

## 6. Final 13-step planner investigation

All large comparisons used four GPUs with batch 16 per GPU unless a smaller learnability check was explicitly sufficient. Fixed data, checkpoints, masks, seeds and frozen-component audits controlled the comparisons.

| Step | Experiment | Finding / disposition |
|---|---|---|
| 1 | Compare original B prior and conditional descendants | Original prior was already weak: late-region hidden-unit accuracy 1.44% under its null condition. Evidence did not support loss of a previously strong prior as the sole problem |
| 2 | Tiny B-only reconstruction, 16 recordings, 600 updates | Reached 100% on fixed and fresh half-masks of those same training recordings. Existing architecture can memorize this task |
| 3 | Targets, units, normalization, padding, frequencies and overlap audit | Sampled 64 train/128 development caches; 576 independent nearest-center checks found zero discrepancies. No sampled target/index bug or unit collapse. Template concentration remained |
| 4 | Matched 640-update learning-rate/masking recipes | Random-half-gap accuracy rose about 25% → 61% with LR 0.0001 → 0.001. Balanced weighting added no clear benefit over original weighting at the higher LR |
| 5 | Optional simpler denoiser | Gate-deferred after existing-model learnability succeeded; it was not necessary to run every optional branch |
| 6 | Expanded B prior, 3,072 examples, 3,200 additional updates | Random-half-gap accuracy 65.85%; contiguous-half-gap only 4.51%. Better local reconstruction did not establish whole-response planning |
| 7 | Paired tiny learning and broader conditional recipes | Tiny training pairs were memorized; broad balanced/full-mask/rehearsal arms still produced poor held-out A-only audio |
| 8 | Actual A-only sampled-audio loss, frozen evaluator, matched CE control | Real gradients and frozen boundaries verified. Fresh-case WER 113.83% with CTC versus 113.78% CE control; no confirmed benefit in this pilot |
| 9 | Acoustic robustness to correct/corrupted/predicted units | Better units improved audio through fixed acoustics. No new acoustic fine-tuning was selected |
| 10 | Predicted versus true B duration | Original eight-case duration MAE about 0.774 seconds. True B duration did not reliably rescue response content |
| 11 | Predicted/zero/shuffled affect | Changed pitch and generated content, but did not establish appropriate meaning or validated control of all six dimensions |
| 12 | Sampling, generated hints, smaller vocabulary, run compression, causal branch | None produced a reliable A-only response candidate; detailed findings below |
| 13 | Three training RNG seeds, fresh development cases, matched audio, integrity and listening preparation | Reconstruction improvement confirmed; free-response quality remained poor. Human ratings pending; final test deferred |

### Reconstruction improvement was real but narrower than response generation

On the same fresh 32 development conversations, random-half-gap reconstruction improved consistently across the three training RNG seeds:

| Training RNG seed | LR 0.0001 | LR 0.001 |
|---|---:|---:|
| 42 | 25.49% unit accuracy | 60.53% |
| 43 | 25.70% | 60.60% |
| 44 | 25.42% | 60.50% |

All arms shared the same original initializer. These are not three independent pretrained initializations or 96 independent evaluation conversations. Contiguous-half-gap accuracy remained only about 4.2–4.6% in the higher-LR arms.

On eight fresh audio cases with supplied true B hints and true B duration, iterative reconstruction WER improved **42.47% → 11.79%**; direct first-pass reconstruction improved **42.45% → 12.21%**. Nearest-visible copying gave 27.89%, full correct B units 10.25%, and original recordings 2.21%. These are B-assisted diagnostics, not ordinary responses generated from A.

### Conditional training and the sampled-audio experiment

The tiny paired experiment could reproduce training pairs and showed sensitivity to A, but held-out quality remained poor. On the broader 2,048-example selection, balanced masking, fully hidden targets and B-only rehearsal all failed to establish good free replies. Shuffling A changed outputs, which demonstrates dependence, not correct understanding.

The sampled-audio planner comparison used 160 matched updates at LR 0.00003, starting from the same balanced conditional checkpoint. The additional frozen CTC term had weight 0.02 and used one rotating waveform per rank every four updates. Predicted duration and actual A-only sampled audio were used. B text was a training loss target, not an inference input.

Gradients passed through frozen acoustics, the codec and the recognizer to a biased final-step straight-through unit estimator. Numerical checks did not indicate that rounding errors caused the semantic failure. The absent fresh-case gain does not rule out every possible evaluator, weight, training budget or gradient estimator.

### Representation and decoding branches

| Branch | Result | Interpretation |
|---|---|---|
| Greedy, categorical and random-remask sampling | No reliable response candidate | Different sampling alone did not fix content planning |
| Generated B hints | Random-gap accuracy 66.18% → 61.92%; contiguous-gap 5.76% → 4.93% | Error propagation exists; long-range prediction was already weak with true hints |
| 256-unit vocabulary | Correct-unit WER 34.88% versus 12.66% with 1,024 units on the same original eight cases | Rejected as a drop-in replacement under these frozen acoustics; does not disprove all smaller-vocabulary systems |
| Exact repeated-unit compression | About 31% shorter; 50 Hz → about 34.5 runs/second | Not equivalent to a 12.5Hz semantic representation; duration factorization deferred |
| Causal planner | Same 1,024 units, width 256 and four blocks; 1,600 extra prior + 1,600 conditional updates | Teacher-forced prediction improved, free replies remained incoherent; extra compute prevents claiming a matched architectural win |

The causal model predicts each B unit from A plus its own previously generated units. Training uses shifted ground-truth prefixes. Causal attention, fixed total-length conditioning, reproducible sampling, exact resume and A-only inference boundaries were tested. Fresh development teacher-forced unit accuracy was about 54.23%; free-running correct-prefix accuracy was only about 2.60% versus 0.97% with shuffled A on the sampled free-running check. These figures measure different tasks.

### Final matched A-only audio comparison

All five candidates responded to the same eight fresh development inputs. None hit the ASR token limit in this comparison.

| Candidate | Mean reference WER |
|---|---:|
| Balanced masked planner | 111.73% |
| Matched additional CE control | 113.78% |
| Matched sampled-audio CTC planner | 113.83% |
| Causal planner, greedy | 115.95% |
| Causal planner, categorical | 108.21% |

The categorical causal score was numerically lowest in this small comparison, but it did not establish reliable response quality or a clear advantage over the balanced masked planner. Several fitting openings still led to unrelated continuations. Some legitimate alternative short replies had high reference WER, reinforcing that WER alone is insufficient.

## 7. Last two regenerated examples

The last inspected response checkpoint was causal conditional step 1,600, SHA-256 `157abc2b5ffe5222113f6f0c591a3685e62bb1e7ee6a0c17356995d1819fbf51`.

Sampling used temperature 0.8, top-k 20, planner seed 42 and acoustic seed 42. The first two entries of the predefined fresh audio selection were used; examples were not chosen based on attractive output. Regenerated WAVs were byte-identical to their previous experiment outputs, and all model/codec weights remained unchanged.

| A situation | Generated-response observation from ASR | Duration / reference WER |
|---|---|---|
| A was pleased that a younger brother cooked a good first meal | Garbled congratulatory opening followed by unrelated wording | 8.33 seconds / 181.82% |
| A was frightened by finding a snake in a bathroom | Began by saying A sounded excited, followed by incoherent wording | 6.47 seconds / 100% |

Original B recordings were recognized with 0% WER for both examples. Generated waveforms were finite and had no raw clipping. This confirms basic signal validity, not naturalness or empathy. These were repeat generations of already evaluated development cases, not new independent evidence. No direct human listening score is claimed.

## 8. Verification and limitations

- Final planner investigation: 71 saved endpoint integrity audits passed. Protected initializer/checkpoint and selection/manifest hashes remained unchanged.
- Pre-release development suite: 281 tests ran, 278 passed and three opt-in tests were skipped. Package-specific rerun results are recorded separately in `RELEASE_CHECKS.md`.
- Checks cover dimensions, padding invariance, losses, gradients, checkpoint identity, strict loading, masked/causal sampling, exact resume, distributed evaluation partitioning, selection integrity, A-only boundaries and report interpretation.
- Four-GPU save/resume and frozen-component checks were executed during the research runs. Creating this ZIP did not rerun GPU training.
- Forty anonymized listening clips were prepared for five candidates on the same eight inputs. Preparation is not completed human evaluation; ratings remain blank. The rater package supplies A text, not A audio.
- Final-test audio/features were not evaluated in the final planner investigation. Manifest/split metadata may have been read. No claim is made that every test-related metadata record was never accessed.
- Target-integrity audits were sampled, not a proof that every source record is correct. Lexical duplicate filters cannot rule out every semantic paraphrase.
- Research results are limited to the tested selections, styles, voice groups, seeds and budgets. Better reconstruction does not imply open-ended conversational competence.

## 9. Remaining problems and proposed next work

| Remaining problem | Next justified direction | Implemented in this release? |
|---|---|---|
| Coherent, emotionally appropriate A-to-B content does not generalize reliably | First test a reviewed training-only response-selection baseline with held-out A inputs and unsuitable candidate controls | No; proposed after the completed experiments |
| Detailed 50Hz prediction is being asked to carry response-level content planning | If selection succeeds, investigate a shorter content plan followed by detailed speech realization | No new hierarchy implemented |
| Highly concentrated reply templates and uncertain emotion labels | Review representative A/B pairs and balance wording/context coverage | Audits implemented; a new curated corpus has not been produced |
| Six affect controls are not all semantically validated | Obtain trustworthy emotional supervision and run controlled human/measurement tests | Measured prosody targets and ablations implemented; missing labels remain missing |
| Human response quality remains unmeasured | Complete blinded intelligibility, relevance, emotional appropriateness and naturalness ratings | Package prepared; human ratings pending |
| Final generalization claim cannot be made | Select a justified candidate using development evidence before opening the final test | No candidate qualified; final-test evaluation deferred |

The next proposed response-selection baseline would be LLM-free but have a restricted response vocabulary. It would be an intermediate diagnostic, not a silent replacement for the original goal of generating novel replies. No such planner, newly curated dataset or additional long training run is claimed as part of this snapshot.

## 10. Release cleanup

The GitHub ZIP contains one current source tree and this consolidated history. Model, loss, sampler, data and test source are copied without changing the trained architecture. Internal checkpoint identifiers and research-mode implementations remain compatible with saved artifacts. Old launch queues, the hard-coded two-example replay helper, obsolete mel-only entry points, duplicate status/readme documents, raw records, generated media and training artifacts are excluded. `.gitignore` covers runtime data and credentials while allowing documentation to be committed.

This source release is not a checkpoint release. The historical private initialization, codebook, manifests and selections are required for exact reruns. The preserved code and report explain the recipes and their outcomes without redistributing the dataset or claiming that random weights reproduce the measured results.
