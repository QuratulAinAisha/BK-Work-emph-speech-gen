# Source attribution and bundled notices

This project extends the user-supplied `2026BK-team1-integrated-baseline.zip`, which includes PerFRDiff-derived components. The supplied root MIT license and its copyright attribution are preserved in [LICENSE](LICENSE).

| Included component | Notice retained |
|---|---|
| Original integrated baseline and derived project source | Root `LICENSE` |
| FaceVerse support source and small normalization/reference arrays | `external/FaceVerse/LICENSE` |
| PIRender support source | `external/PIRender/LICENSE.md` (Attribution-NonCommercial 4.0 International) |

Component-specific notices remain applicable; the root license does not replace them. The inherited facial-rendering support is optional and not used by the current speech waveform path. Large external face models and renderer weights are not bundled.

HuBERT (`facebook/hubert-base-ls960`), EnCodec (`facebook/encodec_32khz`), the frozen wav2vec2 content evaluator (`facebook/wav2vec2-base-960h`) and the independent Whisper evaluator (`openai/whisper-base.en`) are external pretrained models. Their weights and datasets are not redistributed here. Runtime model IDs and pinned revisions, where used, are recorded in source/configuration. The experiment dataset, generated media and private model checkpoints are excluded from this source release.
