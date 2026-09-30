# Broader-data readiness check — September 29, 2026

The broader data selection is prepared. The current speech-unit interface did not
pass the predeclared acoustic reconstruction gate on the broader controls. No new
planner training or 100-epoch run was started by this check. All evaluation GPU
processes have finished.

## Data audit

- The original manifest contains 10,000 distinct conversations, with no conversation
  ID crossing the original train/validation/test boundaries.
- A lexical audit flagged 65 cross-split pairs with identical or closely overlapping
  input wording. Different IDs alone were therefore insufficient.
- Excluded 51 affected training conversations from the new training candidates.
- Also excluded validation candidates related to any original training conversation,
  because existing checkpoints retain that earlier training history. This removed
  three candidates from the first proposed 128-example selection.
- Final selection: **2,048 distinct training conversations and 128 validation
  conversations**, one style and unverified voice group, with responses lasting 3–8 s.
- The final validation selection has no overlap with the previous 24-example pilot
  and no detected lexical overlap with the original training split.
- Forty-seven of the original 64 pilot training conversations occur in the expanded
  training selection; this is allowed training-to-training overlap.

The lexical rule is exact normalized input text, or word-bigram Jaccard similarity
at least 0.8, at least six words, and length ratio at least 0.8. It does not detect
all semantic paraphrases. Original source files and splits were not changed. Final
test metadata was used only to exclude leakage; test audio and test metrics were
not used for model selection.

## Broader acoustic controls

Thirty-two validation conversations were selected in advance from the corrected
selection and evaluated across four GPUs. Each used the same continuous acoustic
checkpoint, sampling seed, requested style, voice group and reference duration.

| Path on the same 32 examples | Mean reference word error rate |
|---|---:|
| Original B recording | 4.50% |
| Original B codec features → fixed decoder | 5.80% |
| Correct continuous B features → audio generator | 14.88% |
| Correct B units from the current 1,024-unit codebook → audio generator | 18.98% |

These are **oracle reconstruction checks**, not free A-to-B response tests. Correct
B features are deliberately supplied to isolate the acoustic interface. Whisper
base.en supplies the diagnostic transcription; no human listening scores are claimed.

The declared gate requires mean oracle error <=20%, excess error over the original
recording <=10 percentage points, and at least 75% of examples within 20 points of
their reference control. Continuous features narrowly miss the excess-error limit:
10.37 points, with 26/32 cases meeting the individual criterion. Units have 14.48
points of excess error and 22/32 cases meeting that criterion, failing both of those
requirements. The continuous result is borderline; these small development samples
do not establish population-level performance.

The unit conversion adds approximately 4.10 percentage points of mean error in this
matched sample. For example, the target phrase about “Croatia beat the hosts” was
recognizable with continuous features but became substantially garbled after unit
conversion. This shows why the pilot's eight-example reconstruction success was not
enough to approve broad deployment.

## Decision

The data is ready for a broader experiment, but the audio interface needs attention
before using generated speech to judge planner generalization. The current codebook
was fitted to only 64 training conversations. The next candidate is a codebook fitted
to representative speech from the expanded **training** selection, followed by the
same controlled reconstruction checks. The acoustic generator also needs broader
validation/adaptation if its reconstruction remains borderline. A replacement
codebook changes unit meanings and must be matched to the planner checkpoint.

Only after those controls pass should the bounded broader planner run proceed with
acoustics initially frozen. Passing these checks would establish readiness for that
experiment, not guarantee an appropriate response to unseen situations.

## Artifacts

Server: `/home/aisha/bk-full-architecture/outputs/generalization_check_clean`.

Local copies: `outputs/server_training_results/generalization_check`.

- `selection.json`: fixed 2,048/128 selection and conversation IDs.
- `data_audit.json`: exclusions and duplicate-check criteria.
- `controls_rank0.json` through `controls_rank3.json`: predeclared acoustic samples.
- `oracle_rank0/report.json` through `oracle_rank3/report.json`: per-example results.
- `result.json`: combined gates and reconstruction measurements.

The first draft selection and its diagnostics remain on the server under
`outputs/generalization_check` for traceability. Only the corrected selection is
reported above. The codebooks, old checkpoints and original dataset were preserved.
