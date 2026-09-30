"""Single-process and torchrun execution. / 단일 프로세스와 torchrun 실행."""

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Sampler


LOSS_NAMES = ("affect", "duration", "semantic", "codec_flow", "total")


class LossForward(nn.Module):
    """Keep DDP's forward hooks active. / DDP의 forward 훅을 유지합니다."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.losses(batch)


class ExactDistributedEvalSampler(Sampler):
    """Shard validation without duplicated samples. / 중복 없이 검증 데이터를 분할합니다."""

    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


@dataclass
class DistributedRuntime:
    device: torch.device
    rank: int = 0
    world_size: int = 1

    @classmethod
    def initialize(cls, requested_device):
        world = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        device = torch.device(requested_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this Python environment")
            if world > 1:
                device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            elif device.index is None:
                device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
        if world > 1:
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=60),
                                    **({"device_id": device} if device.type == "cuda" else {}))
        return cls(device, rank, world)

    @property
    def primary(self):
        return self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    def mean_losses(self, sums, count):
        values = torch.tensor([sums.get(name, 0.0) for name in LOSS_NAMES] + [count],
                              device=self.device, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(values)
        if values[-1] == 0:
            raise ValueError("No samples were processed")
        return {name: float(values[index] / values[-1]) for index, name in enumerate(LOSS_NAMES)}

    def gather_rng(self, generator):
        state = {"loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        if not self.distributed:
            return [state]
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
