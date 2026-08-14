
from __future__ import annotations

import random
from typing import Any, Iterator

from torch.utils.data import Dataset, IterableDataset


class LazyDataLoaderIterator:

    def __init__(self, dataloader: Any):
        self.dataloader = dataloader
        self.iterator: Iterator[Any] | None = None

    def __next__(self) -> Any:
        if self.iterator is None:
            self.iterator = iter(self.dataloader)
        return next(self.iterator)


class MixedDatasetSampler:

    def __init__(
        self,
        datasets: dict[str, Dataset | IterableDataset],
        weights: dict[str, float],
        iterators: dict[str, Iterator[Any]],
        seed: int | None = None,
    ):
        self.datasets = datasets
        self.weights = weights
        self.iterators = iterators
        self._rng = random.Random(seed)

        if len(weights) == 0:
            raise ValueError("weights is empty")
        for name, w in weights.items():
            if float(w) <= 0.0:
                raise ValueError(f"Dataset weight must be > 0, got {name}={float(w)}")
            if name not in datasets:
                raise ValueError(f"Unknown dataset in weights: {name}")
            if name not in iterators:
                raise ValueError(f"Missing iterator for dataset: {name}")

        total_weight = float(sum(float(v) for v in weights.values()))
        self.probs = {name: float(w) / total_weight for name, w in weights.items()}
        self.dataset_names = list(datasets.keys())
        self.prob_list = [self.probs[name] for name in self.dataset_names]

    def sample(self) -> tuple[str, Any]:
        dataset_name = self.choose_dataset_name()
        batch = self.next_batch(dataset_name)
        return dataset_name, batch

    def choose_dataset_name(self, allowed_dataset_names: list[str] | None = None) -> str:
        if allowed_dataset_names is None:
            names = self.dataset_names
            probs = self.prob_list
        else:
            names = [name for name in self.dataset_names if name in set(allowed_dataset_names)]
            if len(names) == 0:
                raise ValueError("No allowed dataset names available for sampling.")
            probs = [self.probs[name] for name in names]
        return self._rng.choices(names, weights=probs, k=1)[0]

    def next_batch(self, dataset_name: str) -> Any:
        if dataset_name not in self.iterators:
            raise ValueError(f"Unknown dataset iterator: {dataset_name}")
        try:
            batch = next(self.iterators[dataset_name])
        except StopIteration as exc:
            raise StopIteration(f"Dataset {dataset_name} exhausted") from exc
        return batch

    def get_sampling_stats(self) -> dict[str, float]:
        return {
            "probabilities": self.probs.copy(),
            "sampling": self.weights.copy(),
        }
