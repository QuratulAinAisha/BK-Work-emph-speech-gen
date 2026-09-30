# Affective Response Transport / 정서 반응 전달

The full architecture is now implemented separately: see
[README_full_speech.md](README_full_speech.md). This guide describes the original
standalone module-3 interface, which remains supported.

Module **3** of the supplied architecture is implemented. It connects the existing
Speaker Behavior Encoder (SBE) and Fusion MLP to a lightweight temporal Transformer:

```text
Person A's mel + 3DMM + AU/VA
    -> existing SBE branches -> existing fusion -> context c [B,T,512]
Person A's AU/VA [B,Te,25] -> valid-prefix alignment ---------+
                                                          |
context c -> projection + emotion projection + positions --+
    -> temporal Transformer -> bounded affect head
    -> target affect a_B* [B,T,6] + valid mask [B,T]
```

**No training is run.** The archive supplies no SBE or affect-transport checkpoints.
The synthetic demo executes the real modules with randomly initialized weights to
verify wiring, shapes, and file outputs. Those values are **not meaningful or
validated empathetic predictions**. Freezing or evaluating random weights does
not make them pretrained. Loading a checkpoint also does not establish its quality.

**학습은 실행하지 않습니다.** 데모는 미학습 가중치로 연결과 입출력을 검증합니다.
출력값은 검증된 공감 예측이나 음성이 아닙니다.

## Architecture scope

| Diagram component | Status in this change |
|---|---|
| 1. Speaker encoding | Existing implementation reused without changing its weights or source |
| 2. Fusion MLP | Existing implementation reused; returns `c` and its valid mask |
| 3. Affective Response Transport | Implemented: projections, temporal positions, Transformer, six control outputs |
| 4. Response length predictor | Future consumer of `context`, `affect`, and `affect_summary`; not implemented |
| 5. Semantic response planner | Future consumer of the full affect trajectory; not implemented |
| 6. Codec latent generation | Future consumer of the full affect trajectory; not implemented |
| 7. Neural codec decoder | Not added in this module-only change |

The existing `train.py`, `infer.py`, mel decoder, datasets, and legacy models remain
available. `infer.py` still runs the original mel-generation baseline; use
`infer_affect.py` for this new path. Affect is not silently inserted into the old
decoder, whose input contract does not include this trajectory. No LLM, training
loop, loss, optimizer, duration model, or audio-generation claim is added.

`model/__init__.py` now loads exports on demand. All original export names remain;
using module 3 no longer imports unrelated legacy diffusion dependencies.

## Run now

From the repository root, the prepared local environment can run:

```powershell
.\.venv\Scripts\python.exe infer_affect.py --demo --output outputs/affect_demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On another machine, use Python 3.10+ and install the minimal dependencies:

```bash
python -m venv .venv
# Activate your environment first. / 먼저 가상 환경을 활성화합니다.
python -m pip install -r requirements/affect.txt
python infer_affect.py --demo --output outputs/affect_demo
```

The default demo passes two synthetic Person-A samples through the **actual SBE,
Fusion MLP, and transport module**, in evaluation/inference mode. Expected outputs:

| Field | Shape / meaning |
|---|---|
| `context` | `[2,11,512]`, existing fused context |
| `context_mask` | `[2,11]`, `True` means valid |
| `affect` | `[2,11,6]`, target affect `a_B*` |
| `affect_mask` | `[2,11]`, same validity as context |
| `affect_lengths` | `[11,7]`, valid frames per sample |
| `affect_summary` | `[2,6]`, mean over valid frames only |

The output directory contains `affect.npz` (all tensor fields plus feature names),
`affect.csv` (valid frames only), and `summary.json` (shapes, configuration, normalized
ranges, checkpoint paths, and untrained-component status). Use a different
`--output` directory to keep previous runs; the three filenames are reused.

## Affect contract

The six channels have a fixed, checkpoint-validated order:

| Index | Name | Korean | Valid range |
|---|---|---|---|
| 0 | `valence` | 정서가 | `[-1,1]` |
| 1 | `arousal` | 각성도 | `[0,1]` |
| 2 | `pitch` | 음높이 | `[0,1]` |
| 3 | `energy` | 에너지 | `[0,1]` |
| 4 | `speaking_rate` | 발화 속도 | `[0,1]` |
| 5 | `dominance` | 주도성 | `[0,1]` |

The diagram leaves the extra affect channels open; dominance is the chosen sixth
channel. These are continuous **normalized controls**, not Hz, dB, or syllables per
second. Downstream physical-unit conversion requires agreed dataset statistics.
Target definitions and normalization must match these ranges when weights become
available. Person A's AU/VA input is a feature vector, not an emotion class ID;
no undocumented AU column is assumed to be valence or arousal.

The default network is two Transformer layers with 128 hidden units, four attention
heads, and a 256-unit feed-forward network. `configs/affective_transport.json`
contains the settings. Attention is bidirectional over the observed input turn;
this is not a streaming/causal model. New code comments and docstrings use short
English and Korean explanations.

## Python integration

```python
import torch
from model import AffectiveResponseTransport, SBEWithAffectiveTransport
from model.speaker_behavior_encoder import SpeakerBehaviorEncoder

# Random weights for wiring only. / 연결 확인용 미학습 가중치입니다.
sbe = SpeakerBehaviorEncoder()
transport = AffectiveResponseTransport()
pipeline = SBEWithAffectiveTransport(sbe, transport).eval()

# Use the existing collated dataset fields. / 기존 배치 필드를 사용합니다.
with torch.inference_mode():
    result = pipeline(
        batch["mel_in"], batch["dmm_in"], batch["au_in"],
        mel_len=batch["mel_in_len"],
        dmm_len=batch["dmm_in_len"],
        au_len=batch["au_in_len"],
    )
    a_star = result["affect"]  # [B,T,6] / B의 목표 정서 궤적
```

For callers that already have fused context:

```python
with torch.inference_mode():
    output = transport(context, speaker_emotion,
                       context_mask=context_mask, emotion_lengths=emotion_lengths)
    a_star = output.trajectory
    affect_summary = output.mean_pool()
```

Inputs must be floating tensors on the same device, with the same dtype and batch
size. `context` is `[B,T,512]`; `speaker_emotion` is `[B,Te,25]`. A provided
`context_mask` must be boolean `[B,T]`, with a nonempty valid prefix in every row.
Omitting it marks every context frame valid. `emotion_lengths` is integer `[B]`;
**omitting it treats all `Te` frames as valid**, even when context has padding.
Always pass the actual emotion lengths for padded batches. Empty sequences,
invalid lengths, interior mask gaps, and non-finite valid values raise errors.
Padded values, including NaNs, are ignored and output padding is exactly zero.

## Time and input compatibility

The wrapper deliberately follows the existing SBE contract: 80D mel, **486D 3DMM**,
and 25D AU/VA. The diagram's 58D 3DMM is not silently substituted for 486D input.
The baseline uses mel frames at `22050/256` Hz, video at 25 Hz, and context at 25 Hz,
which differs from the diagram's example 16 kHz waveform and 100 Hz mel frame rate.
This feature-level module does not extract features from raw video/audio or change
those upstream assumptions.

SBE determines valid context lengths as
`floor(min(mel_len / mel_hz, dmm_len / video_hz, au_len / video_hz) * target_hz)`,
with its existing one-frame minimum. Each valid emotion prefix is linearly resized
to its sample's valid context length with `align_corners=False`, matching the SBE
fusion's existing interpolation convention. The wrapper rejects zero input lengths
before SBE can clamp them. Alignment follows **normalized valid-prefix time**, not
timestamp-based cropping; inputs should describe synchronized clips of the same
turn. Unequal modality durations inherit the baseline's time-warping convention.

`a_B*` remains on this context grid. It does not predict B's response duration and
is not already at the future codec's 50 Hz. Modules 4-6 must determine response
length and align the trajectory to the response/codec grid when implemented.

If your inputs really use 58D 3DMM or 100 Hz mel, the SBE API supports explicit
`dmm_dim` / `mel_frame_hz` settings. Pass them using `--sbe-config settings.json`,
with compatible SBE weights. Changing feature dimensions is not checkpoint-compatible.

## Files and checkpoint inference

`--input` accepts an NPZ with exactly one complete route (floating, batched features):

| Route | Required keys | Optional padding keys |
|---|---|---|
| Cached context | `context`, `speaker_emotion` | `context_mask`, `emotion_lengths` |
| Person-A features | `mel`, `dmm`, `au` | `mel_len`, `dmm_len`, `au_len` |

Export these arrays with `np.savez`; object/pickled arrays are not accepted.
Cached context has no assumed frame rate (`context_frame_hz: null` in its report).

```bash
# Explicit untrained integration check. / 명시적인 미학습 연결 검사.
python infer_affect.py --input features.npz --allow-untrained --output outputs/affect_check

# Use real weights when available. / 실제 가중치가 준비되면 사용합니다.
python infer_affect.py --input features.npz --checkpoint checkpoints/affect.pt --sbe-checkpoint checkpoints/stage2.pth
python infer_affect.py --input cached_context.npz --checkpoint checkpoints/affect.pt
```

Transport checkpoints are saved with `transport.save_checkpoint(path)` and loaded
with `AffectiveResponseTransport.from_checkpoint(path)`. They include the config,
feature order, format version, and state dict. Saving a checkpoint does not train
the model or certify its weights. Loading is strict and uses `weights_only=True`.
The SBE loader accepts a full standalone SBE state dict or the `sbe.*` weights from
the existing Stage-2 checkpoint, including an optional `module.` prefix. Partial
SBE branch checkpoints are rejected. For non-default SBE architectures, supply the
matching `--sbe-config` as well. A missing checkpoint never silently falls back to
random weights; `--demo` or `--allow-untrained` is required.

## Verification

Run `python -m unittest discover -s tests -v`. Tests cover original SBE/split behavior,
real SBE-to-transport integration, conditioning on both inputs, temporal attention,
position sensitivity, shape/range contracts, unequal lengths, batch isolation,
padding invariance, malformed input, checkpoint round trips, and NPZ/CSV/JSON export.
All checks use forward inference; no optimizer step or parameter update is performed.
These tests verify implementation contracts, not learned empathy or speech quality.
