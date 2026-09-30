# Step-by-step planner repair experiments

The user authorized the complete 13-step investigation on 2026-09-30. This is an adaptive experiment sequence: a later change is run when its prerequisite succeeds or is recorded as an explicit diagnostic branch. Finishing a run does not establish a working conversation model. Failed prerequisites and pending human listening must remain visible.

## Final status

The scheduled computational investigation is complete. Reconstruction improved reproducibly, but no candidate established reliable A-only response generation. No experimental checkpoint was promoted. The original data selections and three protected starting checkpoints retain their original hashes. All 71 saved endpoint integrity audits passed; no GPU compute process remained after the runs.

| Step | Outcome |
|---|---|
| 1 | Completed: the original B-only prior was already weak. |
| 2 | Completed: the existing planner memorized the tiny reconstruction task. |
| 3 | Completed: sampled target-integrity checks passed; substantial response-template concentration remains. |
| 4 | Completed: higher learning rate materially improved matched reconstruction. |
| 5 | Gate-deferred: the optional simpler denoiser was unnecessary after learnability succeeded. The separate causal response-planner branch was tested in Step 12. |
| 6 | Completed: expanded prior training and held-out reconstructed audio improved. |
| 7 | Completed: conditional recipes learned training pairs but still produced poor development responses. |
| 8 | Completed: real sampled-audio gradients and the matched frozen-CTC pilot were tested; no confirmed fresh-case benefit. |
| 9 | Completed: acoustic corruption tests showed better unit inputs improve speech; additional acoustic fine-tuning was not selected. |
| 10 | Completed: supplied true duration did not reliably repair content. |
| 11 | Completed: affect changes output, but semantic correctness and unsupported emotional channels remain unvalidated. |
| 12 | Completed: sampling, self-hints, smaller vocabulary and causal-planner feasibility were tested; none qualified for promotion. |
| 13 | Computational confirmation completed: three training RNG seeds, fresh development cases and matched audio. Blinded human ratings remain pending; final-test evaluation is deferred because no response candidate qualified. |

The listening package contains 40 randomized, anonymous clips: five saved candidates responding to the same eight fresh development inputs. Share only `outputs/local_analysis/step13_listening_fresh/rater_only.zip`; the coordinator answer key must remain private until rating is complete. Person A context is text only. Preparing this package is not evidence that human listening has occurred.

## Fixed boundaries

- Server: `/home/aisha/bk-full-architecture`; dataset: `/home/aisha/bk-dataset`.
- Preserve all previous checkpoints and the original test split. Only training and previously used development selections enter initial experiments.
- Large comparisons use GPUs 0–3, batch 16 per GPU. Tiny learnability checks may use one GPU because they have very few examples.
- Keep the original codebook and acoustic modules fixed until the acoustic experiment. Do not promote checkpoints using training loss alone.
- Record checkpoint, manifest, selection and source hashes; use identical masks for comparisons. Compare fixed update endpoints, not independently selected best losses.
- Ordinary response generation receives only A inputs and requested style/voice. Supplied B units and B duration are explicitly labeled diagnostic controls.
- Code comments are short English/Korean pairs. Independent speech recognition is evidence, not a substitute for human listening or an empathy score.

## Ordered investigation

| Step | Experiment | Required evidence / next decision |
|---|---|---|
| 1 | Compare the saved B-only prior with conditional descendants | Matched missing-unit tasks, both task-appropriate and null conditioning; distinguish underlearning from forgetting. |
| 2 | Tiny B-only learning test | Fixed then fresh masks on 8–32 training recordings; inspect optimization if fixed examples cannot be learned. |
| 3 | Target and paired-data audit | Indexing, normalization, padding, gradient boundaries, unit frequencies and response-template concentration; review representative A/B pairs. |
| 4 | Masking/loss curriculum | Matched original and easier-to-harder objectives; separate partial/full masking contributions and retain B reconstruction during conditional fine-tuning if needed. |
| 5 | Conditional simpler-architecture branch | If reconstruction remains weak, compare one architecture change at a time with an unchanged control. |
| 6 | Stronger speech prior | Held-out random/span gaps, run interiors/boundaries, copy comparison and actual reconstructed audio; scale only a learnable recipe. |
| 7 | A-controlled response generation | Small paired learning, correct/shuffled/null A, all B hidden, predicted duration; inspect topical and emotional contradictions. |
| 8 | Sampled-audio objective | Frozen evaluator on the actual generated waveform; verify planner gradients and disclose discrete-gradient approximation; independent evaluation. |
| 9 | Acoustic robustness | Matched correct/corrupted/predicted units; fine-tune only if a measured acoustic sensitivity justifies it. |
| 10 | Length and stopping | Predicted versus diagnostic true duration; count truncation, repeated endings and trailing silence. |
| 11 | Affect controls | Predicted/neutral/shuffled trajectories; measure pitch, energy and speaking rate. Unsupported emotional labels remain unsupported. |
| 12 | Conditional representation branch | Change vocabulary, unit-duration factorization or planner sequence model only if earlier evidence warrants it; revalidate acoustic compatibility. |
| 13 | Independent confirmation | Multiple seeds for promising recipes, untouched final test only after selecting a recipe, and blinded human listening. |

## Initial run plan

Results live in `outputs/planner_repair_v1` on the server and are copied to `outputs/server_training_results/planner_repair_v1` locally. Step 1 compares the original B prior, conditional initializer and fixed fused-memory endpoint. Step 2 first uses 16 training conversations, 300 fixed-mask updates and 300 fresh-mask updates, batch 16, learning rate 0.001, dropout disabled for this diagnostic. It does not measure generalization. Step 3 audits training/development data only. Each transition is recorded with its evidence and a concrete decision before dependent training begins.

The most recent completed memory experiment found no confirmed improvement from native-rate A memory. A nearest-visible B-copy diagnostic substantially beat the learned planner on hinted reconstruction, so the first task is to establish a competent denoiser. These observations motivate the sequence; they do not prove a single root cause.

## Initial measured results

Step 1 completed on 128 development conversations. At 25% randomly hidden units, the last 80% of response time had 1.44% accuracy for the original prior under its trained null condition, 1.73% for the first conditional checkpoint under the same null condition, and 2.12% for the current fused endpoint under null conditioning (3.55% under production A conditioning). The prior was already weak; these results do not support a previously strong prior being lost. Null and production conditioning remain distinct tasks.

Step 2 completed 600 updates on 16 training recordings. Fixed-mask and fresh half-mask accuracy both reached 100%; every frozen-weight audit passed. The existing architecture can learn this small reconstruction task. These are reused training recordings, so this does not establish generalization. The optional simpler-architecture branch is implemented but is not justified by this learnability result alone.

Step 3 passed all sampled integrity checks on 64 training and 128 development caches, including 576 independent float 64 nearest-center checks. No sampled duration/length anomalies were flagged. Transcript statistics covered all 2,048 training and 128 development records. The top three two-word response prefixes cover about 62% of training replies; one development B transcript exactly matches a training B transcript, while exact A and A/B-pair overlap was zero. Some B responses infer feelings that are not explicit in A's transcript; audio/video and human review are needed to judge those labels. The unit vocabulary is not collapsed, and no test examples were opened.

Step 4 compares four fixed 640-update endpoints: original masking at LR 0.0001, original masking at LR 0.001, equal-utterance weighting at LR 0.001, and an easier-to-harder random/span curriculum at LR 0.001. All use the identical original prior initializer, selected data, four GPUs and batch 16 per GPU. Scalar losses use different reductions, so the same held-out gap metrics are the comparison. The curriculum jointly changes difficulty and span frequency; it is a recipe comparison, not proof of one isolated mechanism.

## Completed recipe and scale comparison

The primary B-only task uses null A/style, matching prior training. Every recipe receives the same 128 development conversations and mask seeds 42/43. Accuracy below counts only hidden positions and pools frames.

| Fixed endpoint | Random 25% hidden | Random 50% hidden | Random 75% hidden | Contiguous 25% hidden |
|---|---:|---:|---:|---:|
| Original masking, LR 0.0001, 640 updates | 26.89% | 25.14% | 20.43% | 2.03% |
| Original masking, LR 0.001, 640 updates | 65.75% | 61.03% | 48.96% | 5.62% |
| Equal-utterance loss, LR 0.001, 640 updates | 66.11% | 61.17% | 48.65% | 5.19% |
| Random/span curriculum, LR 0.001, 640 updates | 62.91% | 58.05% | 45.71% | 7.95% |
| Expanded prior, 3,200 additional updates | 69.93% | 65.85% | 54.55% | 6.63% |

Higher LR improved random 50% reconstruction by 35.88 percentage points versus the matched lower-LR control (paired conversation bootstrap 95% CI 35.16–36.61). Equal-utterance versus original weighting differed by 0.14 points (CI −0.20–0.47) on this primary task. The interval describes development-case variation, not variation across independently trained models. The simpler original recipe was chosen for scaling; it was not declared universally superior. Curriculum improved long gaps but their absolute accuracy remained low.

Step 5's simpler-architecture branch is implemented but deliberately not run: tiny learnability and broad random-gap reconstruction succeeded with the existing planner. Changing the architecture before investigating conditional generalization would confound the diagnosis.

Step 6 added 1,024 eligible training conversations to the original 2,048, keeping the same 128 development cases. The initial 4,096 request exceeded the eligible pool and failed before training; 3,072 full-batch examples were then selected from 3,135 eligible records. Selection keeps style 0/voice-group 1, durations 3–8 seconds, excludes digit-containing B text and additional A lexical near-duplicates of development cases. It does not establish generalization to other voices/styles. The prior continued from the 640-update original/high-LR endpoint at LR 0.0003 for 3,200 updates. All 16 recipe and 4 expanded-prior frozen-weight audits passed: only the planner changed.

## Actual audio with supplied B hints

These are reconstruction diagnostics on 8 development cases. True B duration and a stated subset of true B units are supplied; none is ordinary A-only generation. The acoustic generator and codec stay fixed, with identical acoustic noise.

| Input to fixed acoustics | Mean reference WER |
|---|---:|
| Original B recording | 0.69% |
| Complete oracle B units | 12.66% |
| Nearest visible copy, 25% hidden | 26.91% |
| Learned first pass, 25% hidden | 15.13% |
| Learned iterative, 25% hidden | 10.00% |
| Random replacement, 25% hidden | 66.18% |
| Similar-unit replacement, 25% hidden | 19.81% |
| Nearest visible copy, 50% hidden | 30.01% |
| Learned first pass, 50% hidden | 14.83% |
| Learned iterative, 50% hidden | 20.08% |
| Random replacement, 50% hidden | 90.29% |
| Similar-unit replacement, 50% hidden | 31.61% |

This also supplies Step 9's initial acoustic corruption comparison. Better predicted units improve the spoken content without changing acoustics. Arbitrary unit corruption is harmful; it does not by itself justify acoustic fine-tuning. Eight cases and automatic transcription do not establish human listening quality. Iteration helps 25% gaps but hurts 50% gaps, motivating a controlled sampler check.

## Paired A-to-B learnability check

Step 7 first used 64 training conversations, 640 updates, LR 0.001, and fully hidden B units, initializing from the expanded prior. Only the planner trained. With oracle B length, eight-pass training unit accuracy was 91.77% for correct A versus 3.14% for shuffled A. First-pass training CE was approximately 0.000096 per frame, consistent with exact memorization before iterative refinement. The mismatch between all-hidden training and partially revealed later inference rounds is a concrete secondary issue to test with one-pass generation.

On 128 development cases, iterative unit accuracy was 3.82% for correct A and 2.58% for shuffled A; first-pass CE was 11.17. Thus poor generalization exists before iterative refinement. Free A-only audio on 8 cases had 116.65% reference WER; supplying true B duration gave 123.93%, while oracle B units gave 12.66%. Duration correction does not rescue this tiny checkpoint. WER against one valid recorded response is only a diagnostic, but these transcripts also require inspection for intelligibility and relevance. The tiny checkpoint is not a production candidate.

The three fixed 1,600-update comparisons used the original 2,048 paired training cases, the expanded prior initializer, LR 0.0003, 4 GPUs × 16 examples, and frozen duration/acoustics: mixed partial/full masking; entirely hidden B; and entirely hidden B plus 0.5-weight B-only partial reconstruction. Their completed results appear below. Correct/shuffled/null-A controls and actual predicted-length audio are evaluated separately. No endpoint is selected by training loss alone.

## Sampled-audio objective and numerical checks

Step 8 is a planner experiment. It generates the actual A-only response using predicted duration, 8 planner passes, 32 acoustic integration steps and the frozen codec, then compares a frozen speech recognizer's logits to the recorded B transcript using CTC. B text enters only after waveform generation. CE remains an anchor on every example; waveform supervision uses one rotating example per GPU every 4 updates at weight 0.02. This gives 4 waveform examples on sampled updates, versus 64 CE examples per update. The discrete choices use a biased final-step straight-through gradient; this is not an exact gradient through argmax or token ranking. Validation waveform coverage differs from training, so scalar validation totals are not an audio-quality selection metric.

Two initial strict numerical preflights failed and remain saved. A rounding fix, `hard + (soft - soft.detach())`, makes the straight-through forward centers exactly equal to the hard centers while preserving the surrogate gradient. After that, semantic vectors, IDs and duration matched inference exactly, but acoustic latent maximum error was 1.57 e-5 against a 1 e-5 raw-unit limit. Rather than repeatedly increasing a tolerance, a fixed-input calibration replayed both execution paths with identical conditioning and noise. Both paths repeated exactly and matched their respective original outputs exactly. With native MHA fastpaths disabled and math attention forced, grad/no-grad latent outputs became bit-identical. The ordinary relative RMS difference was 3.57 e-7; standardized latent maximum difference was 6.79 e-6. Waveform maximum difference was 0.000264 and relative RMS difference 0.000461, within previously established native/cuDNN decoder bounds.

The calibrated real-model preflight passed: finite nonzero gradients reached only the planner; codec/evaluator gradients were absent; replacing B metadata left the generated waveform exactly unchanged. This establishes a valid experimental gradient path, not better speech. The selected conditional checkpoint subsequently passed its own calibration and a 4-rank forward/backward preflight before the matched CE-only versus CE+sampled-CTC pilot.

Local verification at this point: 232 tests passed, 3 skipped. The 23 objective/waveform integration tests also passed under the server's actual PyTorch 2.5.1 runtime. Later tests and experiments are recorded in the evidence ledger as they finish.

## Three conditional endpoints and selection for the waveform pilot

All three arms finished their fixed 1,600 updates (50 passes through 2,048 selected training examples). All saved frozen-component audits passed. Each arm used seed 42; these are recipe comparisons, not multi-seed confirmation.

| Measurement | Mixed masks | Fully hidden B | Fully hidden B + B-only rehearsal |
|---|---:|---:|---:|
| Fully hidden CE, 128 development cases | 7.27 | 8.46 | 8.35 |
| Eight-pass unit accuracy, correct A, 128 cases | 4.54% | 4.02% | 4.08% |
| Eight-pass unit accuracy, shuffled A, 128 cases | 2.73% | 2.20% | 2.04% |
| Random-half reconstruction, null A, 32 matched cases | 58.37% | 16.29% | 62.73% |
| Random-half reconstruction, production A, same 32 | 62.11% | 8.10% | 34.65% |
| Contiguous-half reconstruction, production A, same 32 | 4.73% | 1.49% | 2.56% |
| A-only audio mean reference WER, 8 cases | 103.03% | 100.94% | 147.62%* |

The strong-prior initializer scored 63.90% on the matched 32-case null-A random-half task. Fully hidden conditional training therefore lost much of its learned denoising ability. Rehearsal preserved null-A denoising but did not teach partially visible inputs under A conditioning. Mixed masking best retained the reconstruction behavior used by conditional iterative inference. Its probability estimates on entirely hidden development targets were also better.

*The rehearsal mean includes one recognizer output-limit case. Removing capped cases independently would make the compared conversation sets differ. Matched common-uncapped bootstrap intervals are used instead: mixed minus fully hidden audio WER difference was +2.08 percentage points (95% interval -15.77 to +22.70); mixed minus rehearsal was -1.51 points (interval -12.59 to +9.45). Neither establishes an audio winner. All reviewed ASR transcripts still contain garbling, irrelevant details or sentiment contradictions. Reference WER alone cannot determine whether a different response is valid.

The mixed-mask endpoint (`step07_conditional/balanced/step_1600.pt`) was selected only as the Step 8 experimental initializer. Both Step 8 arms keep mixed-mask CE, no rehearsal, identical initialization and fixed update endpoints; the treatment adds sampled CTC. Its own calibrated parity test passed, followed by finite matching distributed gradient norms on all four ranks. Every rank kept the evaluator frozen and trained only `semantic_planner`. No response-quality claim follows from these execution checks.

## Step 8: completed sampled-audio pilot

Both arms finished 160 additional updates at LR 0.00003, with four GPUs and batch 16 per GPU. They used the same initializer, data order, mixed-mask CE and frozen duration/acoustics. The treatment additionally generated 160 training waveform instances across the four GPUs. Update counts match; computation does not, because waveform generation and its backward pass add work.

| Fixed checkpoint | Fully hidden development CE | A-only audio WER, 8 cases | True-duration audio WER, 8 cases |
|---|---:|---:|---:|
| Initializer | 7.266 | 103.03% | 189.96%* |
| CE continuation | 7.298 | 111.41% | 93.67% |
| CE + actual sampled-audio CTC | 7.296 | 102.06% | 97.86% |

*One initializer true-duration transcript reached the recognizer's output limit. Both new endpoints had zero capped cases in either generated-audio path. These are mean per-case WER values; insertions can make WER exceed 100%.

The treatment's A-only WER was 9.34 percentage points lower than CE continuation, but the paired 95% interval for that reduction was −0.07 to +21.70 points. Its reduction from the initializer was only 0.96 points, with interval −3.64 to +5.77. Both intervals include zero. On the same sparse validation waveform subset, frozen-teacher CTC increased from 4.175 to 4.243. These results do not establish an improvement. The bootstrap describes variation across these eight cases at fixed checkpoints; it does not cover training-seed uncertainty.

The inspected transcripts remain unsuitable. For example, A describes being ignored by a boyfriend on the phone; the sampled-CTC output is recognized as “That sounds wonderful. I see a seller's name.” This is evidence of a continuing content problem, subject to ASR errors. No checkpoint was promoted. This short, sparse, biased-gradient pilot also does not prove that every sampled-audio objective must fail.

Evidence: `step08_waveform/{ce_control,sampled_ctc}/metrics.jsonl` and `audio/report.json`; paired analysis in `outputs/local_analysis/step08_audio_comparison/comparison.json`.

## Steps 10–11: duration and affect controls

These diagnostics use the unchanged mixed-mask initializer on eight development cases. Its duration error averaged 0.774 seconds on these eight cases; the separate 128-case evaluation was 0.966 seconds. Generated files matched the requested sample count in all 48 duration/affect combinations. The baseline ended with low energy in all eight cases and averaged 0.363 seconds of trailing low energy, versus 0.294 seconds in the reference recordings. This verifies timing behavior; low energy does not prove that a spoken sentence is complete.

Supplying true B duration did not reliably repair the content. Its 189.96% mean WER includes one capped recognizer loop, so that number alone must not be interpreted as a large duration-induced audio deterioration. Incorrect or garbled responses persist in the uncapped cases too.

For the affect comparison, A context, requested style/voice, predicted duration and acoustic seed were held fixed. Only the six-channel trajectory was retained, zeroed, or replaced with another A's predicted trajectory.

| Affect trajectory | Mean of per-case voiced median pitch | Mean waveform RMS | Energy-envelope peaks/second | Mean reference WER |
|---|---:|---:|---:|---:|
| Predicted | 261.7 Hz | 0.0609 | 2.373 | 103.03% |
| All zeros | 166.1 Hz | 0.0639 | 2.351 | 97.20% |
| Shuffled | 257.4 Hz | 0.0589 | 2.436 | 99.55% |

The trajectory affects the result, especially measured pitch. This does not show that all six controls have correct meanings or that removing them improves conversation quality. Zeroing is an ablation, not emotional neutrality. Changing affect also changes planned content, so this experiment includes both content and prosody effects. Pitch can be misestimated in poor audio; envelope peaks and ASR word rate are only speaking-rate proxies. Valence, arousal and dominance still lack verified labels. No affect or duration weights were changed.

Evidence: `steps10_11/report.json`, including per-example control curves and waveform measurements.

## Step 12: sampling and self-generated hints

Sampling was retested on the stronger mixed-mask checkpoint. Unit scores use 32 development conversations and true B length, with no B units supplied. Audio uses eight cases with A-predicted duration and the same acoustic noise across variants.

| Sampler | Correct-A unit accuracy | A-only mean WER | ASR-capped cases |
|---|---:|---:|---:|
| Greedy, 8 passes | 4.48% | 103.03% | 0 |
| Greedy, 1 pass | 4.52% | 197.93% | 1 |
| Categorical, 8 passes, seed 42 | 4.40% | 106.07% | 0 |
| Categorical, 8 passes, seed 43 | 4.32% | 103.26% | 0 |
| Random remasking, 8 passes, seed 42 | 4.62% | 105.90% | 0 |
| Random remasking, 8 passes, seed 43 | 4.05% | 103.98% | 0 |

On the seven cases uncapped for every sampler, one-pass WER was 97.64% versus 104.48% for eight-pass greedy. That small, selected diagnostic subset does not establish a quality win. Stochastic sampling reduced adjacent unit repetition toward the reference rate, but did not produce dependable replies. Shuffling A changed about 96–98% of generated units while exact-reference accuracy stayed low: the planner reacts to A, which is different from responding appropriately. No sampler was promoted.

The separate self-hint test keeps a scored region H (25% of B units) hidden throughout. A disjoint region C (another 25%) is first predicted using the remaining half of true B units. H is then predicted with either true C or the model's detached C prediction. This prevents H's answers from leaking into its own prediction.

| H layout, correct A | H accuracy with true C | H accuracy with predicted C | Reduction |
|---|---:|---:|---:|
| Random positions | 66.18% | 61.92% | 4.26 points |
| One contiguous gap | 5.76% | 4.93% | 0.83 points |

These are means over 32 conversations, with two mask seeds averaged within each conversation. Self-generated errors do hurt random-gap reconstruction. However, contiguous reconstruction is already weak with correct hints. Self-hint mismatch alone therefore does not explain the failure to generate a whole response. This diagnostic supplies true B length and partial B content; it is not A-only generation.

Evidence: `step12_sampling/units_report.json`, `step12_sampling/audio/report.json`, and `step12_self_hints/units_report.json`.

## Step 12: smaller vocabulary compatibility

The 256-unit candidate was fitted on 2,048 training conversations and checked before training a new planner against it. The planner was bypassed: all arms received actual B semantic features and true B duration, with the same A conditioning and frozen acoustic generator. Thus this tests how much recorded content the existing acoustic model can recover from each representation.

| Acoustic input | Mean reference WER, 8 cases |
|---|---:|
| Original B recording | 0.694% |
| Existing 1,024-unit representation | 12.662% |
| Candidate 256-unit representation | 34.884% |
| Continuous B semantic features | 6.648% |

Frame-weighted normalized semantic quantization MSE increased from 0.29675 with 1,024 units to 0.44461 with 256 units on the eight evaluation cases. This is error in semantic feature space, not an acoustic training loss. The smaller vocabulary loses information and performs worse as a direct replacement under the existing frozen acoustics. It is not a suitable drop-in change for this run. This does not prove that a 256-unit system cannot work: learning compatible acoustics or a different representation would be a separate experiment. The before/after hashes confirmed that all model, codec and candidate-codebook components stayed unchanged during this evaluation.

Evidence: `step12_vocabulary/audio/report.json`, copied under the same local results root as the other diagnostics.

## Step 12: completed autoregressive feasibility pilot

The isolated autoregressive branch keeps the 1,024-unit vocabulary, four planner blocks, A conditioning, affect, duration and acoustics. Only the planner adapts to predict the next unit from earlier units. Training uses a shifted correct B prefix; ordinary generation uses its own generated prefix and A-predicted duration. Causal-mask, input-isolation, frozen-weight, checkpoint, resume and random-number tests passed before the server runs.

The 1,600-update B-only prior stage reached development CE 1.353 and next-unit accuracy 58.61% on 128 cases supplied with correct B prefixes. In free generation on eight cases, with null A and true B length, prefix-aligned accuracy was 3.81%, unit edit distance per reference unit was 0.946, and adjacent-unit repetition was 35.90%. These are different tasks: predicting after the correct prefix is much easier than building a whole sequence from previous predictions.

The following 1,600 conditional updates also completed. On 128 development cases, teacher-forced CE was 1.896 and next-unit accuracy was 54.90%. On the eight free-unit diagnostic cases, prefix-aligned accuracy was 3.474% with correct A and 3.863% with shuffled A, holding the original predicted length and requested style fixed. This small exact-reference comparison does not show useful response relevance. Those eight unit cases differ from the eight separately preselected audio cases.

| Conditional AR decoding | A-only mean reference WER, 8 audio cases | ASR-capped cases |
|---|---:|---:|
| Greedy | 103.454% | 0 |
| Categorical, temperature 0.8, top 20, seed 42 | 99.797% | 0 |

Neither decoding method produced dependable replies. When A described a boyfriend ignoring them while using the phone, greedy audio was recognized as beginning “That's awesome,” and categorical audio as “That's wonderful.” The remaining wording was also problematic. These are recognizer observations, not human listening judgments, but they do not support a claim of emotionally appropriate conversation. Both evaluation frozen-component audits passed. The small WER difference does not establish a sampler winner.

No AR checkpoint was promoted. The pilot shows that switching to next-unit prediction did not by itself solve response generation under this recipe and budget. It does not establish that every autoregressive approach will fail. Extra prior/conditional adaptation and serial generation also prevent an equal-compute architecture comparison.

Evidence: `step12_ar/{prior_evaluation,conditional_evaluation}/units_report.json`, `step12_ar/conditional_evaluation/audio/report.json`, and `step12_ar/categorical_evaluation/audio/report.json`.

## Step 13: computational confirmation completed

All four new B-only prior runs completed: low LR 0.0001 and high LR 0.001, each with training seeds 43 and 44. Every run started from the same original saved prior, used the original 2,048 training cases and 128 development cases, and stopped at 640 updates with four GPUs and batch 16 per GPU. All 16 frozen-component audits passed across the four saved endpoints per run. Together with the existing seed-42 pair, this tests whether the learning-rate improvement repeats under different training orders, masks and dropout. It does not repeat independent model initializations or demonstrate conversational quality.

A fixed set of 32 fresh development conversations was selected before these evaluations. An immutable snapshot of 1,285 JSON/JSONL reports identified 228 previously exposed validation conversations to exclude. The selector also filters lexical near-duplicates against all 8,500 original training A texts, previous exposed validation A texts, and other newly selected A texts. It preserves original splits and training membership, applies the existing style/voice and 3–8 second duration criteria, and does not rank cases by model output. “Fresh” means outside this recorded detailed exposure; earlier aggregate validation exposure remains, and unrecorded work cannot be excluded by this audit.

The six fixed endpoints have now completed matched reconstruction on these 32 cases. The table uses null A/style, matching B-only prior training, and hides 50% of the units. True B length and the remaining B units are supplied. Accuracy pools hidden frames across the same cases and mask seeds 42/43.

| Missing-unit layout | Training seed | Low-LR accuracy | High-LR accuracy | High minus low, points (95% case interval) |
|---|---:|---:|---:|---:|
| Random | 42 | 25.49% | 60.53% | +35.04 (+33.92 to +36.23) |
| Random | 43 | 25.70% | 60.60% | +34.90 (+33.65 to +36.24) |
| Random | 44 | 25.42% | 60.50% | +35.08 (+33.91 to +36.31) |
| Contiguous | 42 | 2.11% | 4.24% | +2.13 (+1.56 to +2.65) |
| Contiguous | 43 | 2.03% | 4.57% | +2.54 (+1.95 to +3.15) |
| Contiguous | 44 | 2.09% | 4.39% | +2.29 (+1.77 to +2.81) |

The roughly 35-point random-gap gain repeats across all three training seeds on fresh development cases. That supports the higher learning rate for this reconstruction task. Contiguous-gap accuracy remains only 4–5%, despite its smaller positive gain. The experiment confirms better local reconstruction, not whole-response generation.

The intervals use 20,000 paired conversation bootstrap draws. Both learning-rate arms and both mask seeds stay together inside each resampled conversation. Each interval is conditional on one pair of trained checkpoints; it is not a confidence interval across training seeds. The same 32 conversations are reused for all three training seeds, so this is 32 cases, not 96 independent cases. All runs also share one saved initializer.

Evidence: `outputs/local_analysis/step13_seed_confirmation.json`, with report and checkpoint hashes, and `step13_confirmation/seeds/gaps/`.

The locked evaluation plan includes:

- Completed missing-unit tasks on all 32 fresh cases for all six low/high-LR seed endpoints. The separate B-hinted audio evaluation uses the seed-42 pair only, so it is not multi-seed audio confirmation.
- Three fixed masked-planner checkpoints: the mixed-mask initializer, CE continuation, and sampled-CTC continuation. Each receives 32-case controls and eight-case A-only audio evaluation, retaining the matched CE control for the CTC comparison.
- The fixed AR checkpoint with greedy and predeclared categorical decoding, using 32-case teacher/free-unit evaluation and eight-case audio.

Every audio arm uses the same first eight paths from the immutable fresh-32 selection. Sources, checkpoints, selections and completed outputs are hashed; resuming cannot change the protocol or choose a better endpoint after seeing the results. All three masked-checkpoint and both AR evaluations completed, including the 32-case unit/control checks and eight-case audio comparisons. The evidence ledger verified their completion receipts and available local report/audio hashes without reporting remote-only checkpoint bytes as independently rehashed locally.

The fresh B-hinted audio comparison completed for training seed 42. With half of true B units supplied, first-pass WER fell from 42.45% at low LR to 12.21% at high LR; iterative WER fell from 42.47% to 11.79%. Nearest-visible copy scored 27.89%, complete oracle units 10.25%, and the original B recordings 2.21%. Neither compared learned arm hit the recognizer cap. The low-minus-high WER difference was 30.24 points for first-pass reconstruction (95% paired case interval 15.26–44.97) and 30.67 points for iterative reconstruction (12.04–50.13). These eight-case results confirm that the unit-level improvement can improve actual reconstructed audio on fresh development cases. They still supply B hints and true B length and do not establish free-response or multi-seed audio quality. Evidence: `outputs/local_analysis/step13_hinted_audio_comparison.json`.

The separate free-response comparison supplied only A inputs, requested style/voice and predicted duration. All five candidates used the same eight A contexts and recorded B references. None of these 40 recognizer outputs hit its token cap.

| Fresh A-only candidate | Mean reference WER |
|---|---:|
| Mixed-mask initializer | 111.73% |
| CE continuation, 160 updates | 113.78% |
| Sampled-CTC continuation, 160 updates | 113.83% |
| Causal planner, greedy | 115.95% |
| Causal planner, categorical | 108.21% |

The predeclared CE-control-minus-CTC difference was -0.046 percentage points, with a paired case-bootstrap 95% interval from -8.64 to +7.03. This pilot therefore provides no confirmed improvement from sampled-CTC training on the fresh cases. It does not rule out a different training budget, objective weight or gradient estimator. The categorical-versus-greedy AR difference favored categorical reference WER by 7.74 points on these eight cases, but it was one of nine descriptive contrasts without multiple-comparison correction. Categorical versus the mixed-mask initializer remained uncertain (+3.52 points in its favor, interval -14.16 to +24.78). Neither comparison establishes reliable conversation quality.

WER compares against one recorded reply; valid alternative responses can score poorly, and inserted words can push WER above 100%. The actual recognized content also matters. For the account of finding a snake and nearly passing out, both causal outputs began with being “really excited.” Several outputs elsewhere started with a plausible short acknowledgment but then became garbled or unrelated. These are ASR observations, not human listening judgments. All exact transcripts, contexts, per-case differences and uncertainty intervals are preserved in `outputs/local_analysis/step13_fresh_audio_comparison/comparison.md` and its JSON/CSV companions.

A useful counterexample to judging by WER alone is the clean-home input: the mixed-mask model was recognized as saying only “That sounds wonderful.” That is a plausible acknowledgment despite 81.25% reference WER. Also, the ambiguous wish to be Elon Musk does not by itself prove the reference's jealousy interpretation is correct. Independent review of all 40 transcripts found some fitting openings but no candidate with dependable continuation or emotional relevance; this remains a transcript review, not completed human listening.

Across all 32 fresh cases, the causal planner's teacher-forced next-unit accuracy was 54.23% with correct B prefixes. Free greedy prefix-aligned unit accuracy was 2.60% with correct A and 0.97% with shuffled A. The conditioning affects the sequence, but exact-reference unit alignment is not a response-relevance metric. All final AR frozen-component checks passed. No new production candidate was selected.

Latest local verification: 281 tests ran successfully, with 278 passing and 3 skipped. Passing tests and frozen-weight audits establish implementation checks, not speech quality.

Human listening remains pending. The final test split has not been evaluated and remains gated on a selected, justified recipe. No tested checkpoint currently supports a claim of reliable, relevant empathetic speech; completion of these experiments must not be presented as completion of that quality goal.

Evidence: `step13_fresh_v2/confirmation_data_audit.json` and the locked protocols/results under `step13_confirmation`.
