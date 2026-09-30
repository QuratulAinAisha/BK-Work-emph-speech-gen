# Improved baseline: run readiness and completion status

**Superseded September 29, 2026:** quality-v2 was deliberately stopped at epoch 32
after failed generated-response checks. Do not use the old launch instructions below.
The authorized replacement and current findings are in
[README_quality_recovery.md](README_quality_recovery.md). The original historical
snapshot follows for traceability.

Checked on **September 29, 2026, at approximately 06:12 KST** (September 28, 21:12 UTC).
This is a dated snapshot, not a live dashboard.

The quality-v2 implementation, dataset preparation, deployment, and run setup are
complete. The **100-epoch training experiment is still running**. Good generated
speech has not yet been demonstrated.

## What is ready

| Item | Verified state |
|---|---|
| Architecture | Original SBE/fusion, affect transport, duration planning, semantic planning, conditional codec generation, and frozen audio decoding are connected. |
| Quality improvements | Native 768-dimensional HuBERT content features, transcript supervision during training, relevance loss, spectral loss, and staged joint training are implemented. |
| Code comments | New implementation uses short English and Korean comments. |
| Server project | `/home/aisha/bk-full-architecture` |
| Dataset | `/home/aisha/bk-dataset` |
| Prepared examples | All 60,000 records passed preparation validation. |
| Data splits | 51,000 training, 6,000 validation, and 3,000 test responses; no conversation crosses splits. |
| GPUs | Four GPUs, IDs 0, 1, 2, and 3. |
| Batch size | 16 examples per GPU; global batch size 64. |
| Training target | 100 new quality-v2 epochs. |
| Completed stages | Acoustic adaptation (epochs 1–5) and semantic adaptation (epochs 6–10). |
| Current stage | Joint training; 14 epochs completed and epoch 15 underway at this check. |
| Epoch history | Completed epochs are consecutive, without duplicates; recorded losses are finite and world size is four. |
| Checkpointing | Latest checkpoint plus best checkpoints by training stage, with optimizer, scheduler, and per-rank random state for resume. |
| Audio evaluation | Six validation conversations and a controlled six-style sweep every five epochs. |
| Final evaluation | Controller is configured to evaluate 100 unique held-out test conversations and the controlled style sweep. |
| Monitoring | Existing 15-minute monitor remains active for failures, stage changes, completion, and final result collection. |

The original baseline was intentionally stopped after epoch 87 following the user's
quality check. Its checkpoints are preserved. Its former 100-epoch target is cancelled;
the running 100-epoch target belongs to quality-v2.

## Checks already completed

- Focused model checks cover content gradients, staged training, differentiable codec
  loss, invalid CTC alignments, checkpoint round-trip, and exclusion of Person B targets
  from inference.
- A small real-data run passed all three training stages on four GPUs with batch size
  16 per GPU. Checkpoint resume and generated-audio evaluation also passed.
- Existing full-architecture regression checks passed.
- The full prepared dataset passed validation, including conversation-separated splits.
- Nine implementation/test files match the deployed server copies by SHA-256.
- Both latest (epoch 14) and best (epoch 13) checkpoints loaded successfully. All
  400 floating model tensors in each are finite; optimizer, scheduler, and all four
  ranks' random states are present.

These establish software readiness. They do not establish intelligibility, response
relevance, naturalness, or empathy.

## Results so far

At completed epoch 14, training total loss was **1.9427** and validation total loss was
**1.9638**. Recent joint-stage epochs took approximately **17–19 minutes** each.
Future timing can change as the fraction of generated semantic inputs increases.

The latest completed generated-audio evaluation available at this check was epoch 10.
Its automatic transcripts were generic phrases such as “Thank you” and “Thanks for
watching,” rather than convincing responses to Person A. Automatic speech recognition
can also hallucinate these phrases on poor audio. No human listening scores are
claimed, and the run is not being described as a successful speech model yet.

## What remains

1. Finish the remaining joint-training epochs through epoch 100.
2. Complete the scheduled final held-out evaluation.
3. Verify final checkpoint metadata, finite tensors, complete epoch history, and
   the expected evaluation coverage.
4. Collect the final checkpoint, metrics, reports, and representative Person A,
   reference Person B, and generated Person B audio for review.
5. Report measured results and limitations candidly. A decreasing loss is insufficient
   evidence that the system produces good responses.

## Run and recovery

The controller is **already running**. Do not launch a duplicate. If it has stopped,
inspect its recorded error and confirm its old workers have exited before resuming
from the server project directory:

```bash
.venv/bin/python -u scripts/run_quality_training.py --epochs 100 --batch-size 16 --accept-stopped-baseline
```

The controller reuses the validated cache and resumes quality-v2 from its latest
checkpoint when one exists. It must not restart the deliberately stopped baseline.

Live server files:

```text
outputs/bk_quality_v2_100/progress.json
outputs/bk_quality_v2_100/metrics.jsonl
outputs/bk_quality_v2_100/run_status.json
outputs/bk_quality_v2_100/training.log
outputs/bk_quality_v2_100/last.pt
outputs/bk_quality_v2_100/best.pt
```

## Interpretation limits

Generation takes Person A inputs and the requested response style/voice group. It
does not take Person B text, target semantic features, reference audio, or reference
response duration. Transcripts and response targets are training supervision only.

Valence, arousal, and dominance lack direct dataset labels. Male/female metadata
defines voice groups, not verified individual speaker identities. The two Whisper
sizes used for evaluation are related recognizers, and automatic embedding scores
are diagnostics rather than human quality ratings.

See [README_quality_v2.md](README_quality_v2.md) for the dimensions, model changes,
losses, training schedule, and inference command.
