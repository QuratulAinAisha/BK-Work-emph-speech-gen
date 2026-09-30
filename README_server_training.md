# BK server training / BK 서버 학습

Project: `/home/aisha/bk-full-architecture` on `aisha@203.252.206.113`.
Dataset: `/home/aisha/bk-dataset`. The project has its own `.venv`; existing
environments and dataset files are preserved. GPUs **0 and 1** are used.

The launched controller first prepares all targets, then trains **five epochs**,
then generates a held-out test example from `best.pt`. It runs independently of
the SSH connection. The status is in `outputs/bk_5epochs/run_status.json`;
`stage: complete` is written only after training and inference succeed.
At delivery the full job may still be running: check this file, not a static
README, for completion. Do not launch a second copy while it is active.

## Data and first-run choices

- All 60,000 response paths matched after resolving the observed
  `surprised`/`surprise` reference-image filename alias.
- Conversation split: 8,500 train / 1,000 validation / 500 test conversations,
  giving 51,000 / 6,000 / 3,000 responses. Every style of a conversation stays
  in the same split. Audio contents are validated during target preparation.
- Original input dimensions remain mel 80, 3DMM 486, expression 25, context 512.
- No compatible pretrained SBE checkpoint was supplied. SBE trains from scratch,
  including visual/emotion branches. The external HuBERT and EnCodec models are
  pretrained and frozen. This differs from freezing a pretrained SBE in the diagram.
- The six-channel affect tensor is retained. Audio provides log-pitch normalized
  between 60 and 500 Hz, RMS energy normalized from -60 to 0 dBFS, and utterance
  word rate divided by 6 words/second. Autocorrelation pitch is a measured proxy;
  unvoiced frames have zero pitch weight. Word rate is not syllable alignment.
- Valence, arousal and dominance lack documented labels. They have zero direct
  loss weight and learn indirectly through the other response losses. Their
  outputs must not be described as calibrated emotional measurements.
- Voice group 0 is female, 1 is male, from supplied metadata. These are not
  verified individual speaker identities; face-image identities are not used
  as audio identities. True identity control needs a verified audio speaker map.
- This is a generated paired corpus. Five epochs and decreasing training losses
  do not establish intelligibility, empathy or speaker fidelity.

## Commands

Run from the server project directory after target preparation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 .venv/bin/torchrun --standalone --nproc_per_node=4 train_full.py --manifest outputs/bk_prepared/manifest.json --config outputs/bk_source/config.json --train-sbe-from-scratch --device cuda --amp --amp-dtype bfloat16 --epochs 5 --batch-size 16 --workers 4 --output outputs/bk_5epochs
```

Batch size 4 means **4 per GPU**, global batch size 8. Both ranks synchronize
gradients and statistics; rank 0 writes checkpoints. Validation samples are
counted once. Checkpoints contain optimizer, scheduler, scaler and per-rank RNG
state. Training does not use the held-out test responses.

For the complete preparation → training → held-out inference sequence, including
resuming cached preparation and the last completed training epoch:

```bash
.venv/bin/python scripts/run_bk_training.py
```

The initial corpus manifest is reproducible with:

```bash
.venv/bin/python scripts/build_bk_source.py --dataset /home/aisha/bk-dataset --output outputs/bk_source --defer-audio-validation
```

Inspect progress:

```bash
cat outputs/bk_5epochs/run_status.json
cat outputs/bk_prepared/progress_rank0.json outputs/bk_prepared/progress_rank1.json
tail -n 5 outputs/bk_5epochs/metrics.jsonl
```

Output locations:

| Path | Contents |
|---|---|
| `outputs/bk_source/audit.json` | Corpus match report, split counts and limitations |
| `outputs/bk_prepared/` | Resumable target cache and completion marker |
| `outputs/bk_5epochs/preparation.log` | Frozen-teacher extraction on two GPUs |
| `outputs/bk_5epochs/training.log` | Training progress and errors |
| `outputs/bk_5epochs/metrics.jsonl` | Per-epoch aggregate train/validation losses |
| `outputs/bk_5epochs/best.pt`, `last.pt` | Best validation and final/resumable checkpoints |
| `outputs/bk_5epochs/heldout_example/` | Generated WAV, intermediate tensors and reference metadata |
| `outputs/bk_smoke_training/` | Separate successful one-epoch, 20-conversation software check |

Preparation never fabricates missing files or silently truncates responses.
Source/config identity is checked on resume. Atomic sample writes prevent
partially written cache files from being reused. Float32 duration boundaries
are converted to integer audio sample counts before calculating codec lengths.

The temporary SSH private key remains on the client and is excluded from all
archives and server code. Remove only the `codex-bk-temporary-20260928` public-key
entry when remote work is finished; keep other authorized keys.

## Updated GPU plan (2026-09-28)

The user authorized four GPUs for training after the existing two-GPU preparation finishes. Controller PID 576920 adopts preparation PID 447393 without restarting it, then trains five epochs on GPUs 0,1,2,3. Batch size is now 16 per GPU (64 total), explicitly requested by the user; completion must report world_size 4. Earlier two-GPU descriptions in the beginner guide describe the initial plan.
