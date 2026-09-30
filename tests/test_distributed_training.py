"""Two-rank synchronization and resume checks. / 두 프로세스 동기화·재개 검사."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from test_full_speech import make_cache, small_config, sample
from utils.distributed_training import ExactDistributedEvalSampler


class DistributedSamplerTest(unittest.TestCase):
    def test_validation_has_no_duplicates_even_with_empty_rank(self):
        dataset = [0, 1, 2]
        shards = [list(ExactDistributedEvalSampler(dataset, rank, 5)) for rank in range(5)]
        self.assertEqual(sorted(value for shard in shards for value in shard), dataset)
        self.assertEqual(shards[-1], [])


@unittest.skipUnless(os.environ.get("BK_TEST_DDP") == "1", "Set BK_TEST_DDP=1 for two-process training")
class DistributedTrainingTest(unittest.TestCase):
    def test_cpu_resume_matches_continuous_and_shares_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            manifest = make_cache(root, config)
            metadata = json.loads(manifest.read_text())
            for index in (3, 4, 5):
                path = f"sample_{index}.npz"
                np.savez(root / path, **{key: value.numpy() for key, value in sample(index).items()})
                metadata["records"].append({"conversation_id": str(index), "split": "train", "path": path})
            manifest.write_text(json.dumps(metadata))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            common = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                      "train_full.py", "--manifest", str(manifest), "--device", "cpu", "--allow-synthetic", "--batch-size", "1"]
            initial = ["--config", str(config_path), "--train-sbe-from-scratch"]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            def run(arguments):
                process = subprocess.run(common + arguments, cwd=Path(__file__).resolve().parents[1], env=env,
                                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                self.assertEqual(process.returncode, 0, process.stdout[-16000:])
            run(initial + ["--epochs", "1", "--output", str(root / "resumed")])
            run(["--resume", str(root / "resumed/last.pt"), "--epochs", "2", "--output", str(root / "resumed")])
            run(initial + ["--epochs", "2", "--output", str(root / "continuous")])
            a = torch.load(root / "resumed/last.pt", weights_only=True)
            b = torch.load(root / "continuous/last.pt", weights_only=True)
            self.assertEqual(a["world_size"], 2)
            self.assertEqual(len(a["rank_rng_states"]), 2)
            self.assertEqual(a["training_steps"], 4)
            self.assertEqual(len((root / "resumed/metrics.jsonl").read_text().splitlines()), 2)
            for name in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
