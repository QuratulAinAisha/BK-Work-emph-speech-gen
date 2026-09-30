# Checks on the packaged source

Checked on **30 September 2026**, from the staged release directory itself, using CPU execution and offline pretrained-model settings. These checks validate packaging and code behavior; they do not establish conversational audio quality.

| Check | Result |
|---|---|
| Full default suite: `python -m unittest discover -s tests -v` | 281 tests ran in 48.855 seconds; **278 passed, 3 skipped, 0 failed** |
| Python syntax compilation | All 186 included Python source files compiled |
| Public CLI help | 11 entry points returned exit code 0 |
| Module-3 synthetic demo | Passed on CPU; affect shape `[2, 11, 6]`, valid lengths `[11, 7]`; untrained weights explicitly labeled |
| Included Python versus current workspace | Every included Python file is byte-identical |
| Latest causal experiment source provenance | All seven recorded model/trainer/data source hashes match the experiment recipe |
| Original archive comparison | Original SBE, canonical dataset, mel decoder, SBE tests and split tests were already present and unchanged |
| Text scan | No known private server address/account path, local user workspace path, private-key block or recognized access-token pattern found |
| Runtime artifacts | Dataset, trained checkpoints, generated media, logs, environments and caches excluded |
| GPU use for release packaging | None; no new model training or pretrained download |

The three skipped tests are the optional two-process training/resume check and two real-pretrained-model integration checks. They require `BK_TEST_DDP=1` or `BK_TEST_PRETRAINED=1`, respectively, and were not rerun for this packaging task. The historical research runs performed GPU and real-codec checks separately; those are not conflated with this CPU release check.

Environment used for the release check: Python 3.12.14, PyTorch 2.14.0+cpu, Transformers 4.44.2, NumPy 2.3.5, einops 0.8.2, soundfile 0.14.0, SciPy 1.18.1. The earlier GPU experiments used a different environment, described in the README.

The ZIP is checked for readable CRCs and byte-for-byte agreement with the SHA-256 entries in `RELEASE_MANIFEST.json`. That manifest lists every payload file except itself. The separate `.zip.sha256` file identifies the completed archive. Hashes identify this release; ordinary later edits or Git line-ending normalization will change them.

These are the checks actually performed. This is not a claim of a fresh-environment installation, exhaustive security audit, human listening evaluation, or a successful long training run from the public source ZIP alone.
