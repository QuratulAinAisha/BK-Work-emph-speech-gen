# Broader speech-unit experiment

## Completed result: adaptation rejected

The bounded acoustic run completed all five epochs (160 updates) on September 29,
2026, using all four GPUs and batch 16 per GPU. The subsequent 32-case evaluation
failed the declared quality gate, so the controller did **not** start the planner.
The GPUs were released. The older acoustic checkpoint remains preserved.

| Measurement | Before adaptation | After five acoustic epochs |
|---|---:|---:|
| Fixed validation objective, 128 examples | 0.97038 | 0.89081 |
| Reference recording WER, matched 32 examples | 4.50% | 4.50% |
| Correct new-unit reconstruction WER | 15.65% | 24.60% |
| Excess WER over reference | 11.15 percentage points | 20.10 percentage points |
| Cases within 20 points of own reference | 26/32 | 16/32 |

Lower objective loss did not translate into clearer speech. These measurements use
the correct B response units and length, deliberately supplied as a diagnostic;
they do **not** demonstrate generation of a suitable response from A alone.
The new vocabulary improved reconstruction before adaptation, but this adaptation
recipe made it worse. Neither configuration passed the fixed gate.

Checkpoint inspection confirmed that the loss-selected best checkpoint was also
the final step-160 checkpoint, all model tensors were finite, and no unexpected
existing frozen tensors changed. Thus evaluating `last.pt` again would duplicate
the same model. The new planner was initialized for the new unit meanings and
remained untrained; Module 3 and the existing context components stayed unchanged.

The next useful experiment is to improve the acoustic training/selection objective
against actual reconstructed audio before extending planner training. This result
does not justify another 100-epoch run or a claim of improved generalization.
Full resumable checkpoints remain on the server; downloaded reports, audio controls,
codebook and inference checkpoint are under
`outputs/server_training_results/broad_units_v1`.

Authorized after the broader-data readiness check on September 29, 2026. This is a
bounded staged experiment, not an automatic restart of either old 100-epoch run.

## Fixed data and representation

- 2,048 distinct training conversations and 128 validation conversations.
- One response style and unverified voice group; B audio length 3–8 seconds.
- Conversation splits preserved, with lexical duplicate exclusions described in
  `GENERALIZATION_CHECK.md`. Final test audio remains unused.
- New 1,024-center vocabulary fitted to **all 673,910 frames of the selected training
  speech**, with 40 k-means iterations and seed 42. No validation features fit centers.
- Interface remains 768-dimensional normalized HuBERT vectors at 50 Hz. The original
  acoustic normalization is preserved. Lookup centers convert IDs back to this width.
- Codebook SHA-256: `ad95c244b877fa1194be0492e5f8550a02de07521d9b0185ab668a100be2d115`.

New centers have new unit meanings. The old 64-example planner's classifier is not
reused with the new IDs. The broader planner starts with fresh planner parameters;
other model components come from the acoustic initialization.

## Matched initial controls

The acoustic generator stayed unchanged for this comparison on the same 32 validation
examples. Values are mean Whisper base.en reference word error rates.

| Path | WER |
|---|---:|
| Original reference recording | 4.50% |
| Codec reconstruction | 5.80% |
| Correct continuous response features | 14.88% |
| Old units fitted on 64 conversations | 18.98% |
| New units fitted on 2,048 conversations | 15.65% |

The broader vocabulary improved the unit reconstruction control by 3.33 percentage
points, but still missed the fixed excess-error threshold: 11.15 points over the
reference, versus the allowed 10. These are reconstruction controls with B targets
deliberately supplied, not response-planning or empathy scores.

## Bounded stages

Training uses four GPUs (0–3), batch 16 per GPU, global batch 64. There are 32 updates
per epoch, so **five epochs mean 160 optimizer updates**, not 1,200 pilot updates.

1. **Acoustic adaptation:** five epochs on the broader data, learning rate 3e-5,
   warmup/cosine schedule, frozen waveform teacher weight 0.05 every four steps.
   The codec generator and codec content head train; the context, Module 3 and new
   unit planner remain frozen. The externally pretrained codec stays frozen.
2. **Reconstruction evaluation:** 32 predeclared validation examples across four
   GPUs. The actual unit path must pass the same engineering gate: mean oracle
   WER <=20%, excess over references <=10 points, and >=75% of cases within 20 points
   of their own reference controls. The continuous path is a comparator; it is not
   the inference input to the new unit model.
3. **Conditional planner trial:** only if the unit interface passes, train the new
   planner and duration predictor for five epochs at learning rate 1e-4. All acoustic,
   affective and context components stay frozen. Use unit cross-entropy plus duration
   loss, with both fully and partly masked training sequences and eight parallel
   inference refinements. No LLM or intermediate text generation is added.
4. **Evaluate after each planner epoch:** fixed loss evaluation on all 128 validation
   cases, plus A-only audio generation on eight fixed validation conversations.
   Save/resume at epoch boundaries with identical recipe and random-state metadata.
   Review the actual responses before extending training or starting joint updates.

The short run is a learning-curve experiment; it does not promise convergence after
five epochs. A decreasing training loss does not establish generalization.

## Reproduction and monitoring

Server project: `/home/aisha/bk-full-architecture`.

```bash
.venv/bin/python -u scripts/run_broad_units.py
```

Do not launch a duplicate while the controller is active. It takes the existing
recovery lock, requires the fitted vocabulary and initial reconstruction report,
and resumes compatible stage checkpoints. Outputs are under `outputs/broad_units_v1`:

- `run_status.json`, `codebook.pt`, `codebook.json`, `selection.json`.
- `initial/result.json`: new-vocabulary reconstruction controls before adaptation.
- `acoustic/{recipe.json,metrics.jsonl,best.pt,last.pt}`.
- `adapted_oracle/result.json` and `unit_interface_gate.json`.
- If the gate passes: `planner` and `planner_eval_epoch_01` through `05`.

Older checkpoints and codebooks remain preserved. English/Korean comments are used
in the new implementation. The existing monitor follows this bounded experiment;
it must not restart the old full-training controllers or equate completion with
successful conversational speech.
