from __future__ import annotations

import math

import torch
from torch.utils.data import DistributedSampler


class DeterministicVariableLengthDistributedSampler(DistributedSampler):
    """Distributed sampler that emits ``(index, num_frames)`` pairs.

    ``batch_size`` should match the per-rank ``DataLoader(batch_size=...)`` so that
    every local batch step uses a single shared ``num_frames`` across all ranks.
    """

    _NUM_FRAMES_SEED_OFFSET = 1_000_003

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        batch_size=1,
        min_num_frames=4,
        max_num_frames=10,
        num_frames_choices=None,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if num_frames_choices is None:
            num_frames_choices = list(range(int(min_num_frames), int(max_num_frames) + 1))

        normalized_choices = []
        seen = set()
        for value in num_frames_choices:
            value = int(value)
            if value <= 0:
                raise ValueError(f"num_frames choices must be positive, got {value}")
            if value not in seen:
                normalized_choices.append(value)
                seen.add(value)
        if len(normalized_choices) == 0:
            raise ValueError("num_frames_choices must contain at least one value")
        self.num_frames_choices = normalized_choices

    def _build_global_indices(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]

        if len(indices) != self.total_size:
            raise RuntimeError(
                f"Expected padded global index count {self.total_size}, got {len(indices)}"
            )
        return indices

    def _build_local_num_frames(self):
        num_local_steps = math.ceil(self.num_samples / self.batch_size)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + self._NUM_FRAMES_SEED_OFFSET)
        choice_ids = torch.randint(
            low=0,
            high=len(self.num_frames_choices),
            size=(num_local_steps,),
            generator=generator,
        ).tolist()

        local_num_frames = []
        for choice_id in choice_ids:
            num_frames = self.num_frames_choices[choice_id]
            local_num_frames.extend([num_frames] * self.batch_size)
        return local_num_frames[: self.num_samples]

    def __iter__(self):
        global_indices = self._build_global_indices()
        local_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        local_num_frames = self._build_local_num_frames()
        if len(local_indices) != self.num_samples or len(local_num_frames) != self.num_samples:
            raise RuntimeError(
                f"Sampler local size mismatch: indices={len(local_indices)} num_frames={len(local_num_frames)} "
                f"expected={self.num_samples}"
            )
        return iter(zip(local_indices, local_num_frames))


VariableLengthDistributedSampler = DeterministicVariableLengthDistributedSampler


__all__ = [
    "DeterministicVariableLengthDistributedSampler",
    "VariableLengthDistributedSampler",
]
