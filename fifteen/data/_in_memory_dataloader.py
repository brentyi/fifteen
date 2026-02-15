import dataclasses
import math
from typing import Generic, Optional, Sequence, TypeVar

import jax
import numpy as onp

from ._protocols import DataLoaderProtocol

PyTreeType = TypeVar("PyTreeType")


@dataclasses.dataclass(frozen=True)
class InMemoryDataLoader(Generic[PyTreeType], DataLoaderProtocol[PyTreeType]):
    """Simple data loader for in-memory datasets, stored as arrays within a PyTree
    structure.

    The first axis of every array should correspond to the total sample count; each
    sample will therefore be indexable via `jax.tree.map(lambda x: x[i, ...], dataset)`.

    :meth:`minibatches()` can then be used to construct an (optionally shuffled)
    sequence of minibatches."""

    dataset: PyTreeType
    minibatch_size: int

    drop_last: bool = True
    """Drop last minibatch if dataset is not evenly divisible.

    It's usually nice to have minibatches that are the same size: it decreases the
    amount of time (and memory) spent on JIT compilation in JAX and reduces concern of
    noisy gradients from very small batch sizes."""

    sample_count: int = dataclasses.field(init=False)

    def __post_init__(self):
        shapes = [x.shape for x in jax.tree.leaves(self.dataset)]
        assert len(shapes) > 0, "Dataset should contain at least one array."

        sample_counts = [shape[0] for shape in shapes]
        assert all(
            count == sample_counts[0] for count in sample_counts
        ), "All sample counts should be equal."

        object.__setattr__(self, "sample_count", sample_counts[0])

    def minibatch_count(self) -> int:
        """Compute the number of minibatches per epoch."""

        minibatch_count = self.sample_count / self.minibatch_size
        if self.drop_last:
            minibatch_count = math.floor(minibatch_count)
        else:
            minibatch_count = math.ceil(minibatch_count)
        return minibatch_count

    # Note that a Sequence is a SizedIterable with support for index-based access.
    def minibatches(self, shuffle_seed: Optional[int]) -> Sequence[PyTreeType]:
        """Returns an iterable over minibatches for our dataset. Optionally shuffled using
        a random seed."""

        indices = onp.arange(self.sample_count)
        if shuffle_seed is not None:
            onp.random.default_rng(seed=shuffle_seed).shuffle(indices)

        return _Minibatches(
            dataset=self.dataset,
            indices=indices,
            minibatch_size=self.minibatch_size,
            minibatch_count=self.minibatch_count(),
        )


@dataclasses.dataclass(frozen=True)
class _Minibatches(Sequence[PyTreeType], Generic[PyTreeType]):
    """Iterable object for returning minibatches."""

    dataset: PyTreeType
    indices: onp.ndarray  # Shape: (dataset length,)
    minibatch_size: int
    minibatch_count: int

    def __getitem__(self, i):
        if i >= self.minibatch_count or i < -self.minibatch_count:
            raise IndexError()

        i %= self.minibatch_count  # For negative indexing.
        start_index = self.minibatch_size * i
        end_index = min(self.minibatch_size * (i + 1), self.indices.shape[0])
        minibatch_indices = self.indices[start_index:end_index]
        return jax.tree.map(lambda x: x[minibatch_indices, ...], self.dataset)

    def __len__(self):
        return self.minibatch_count


def _check() -> None:
    pytree = [onp.zeros((32, 64, 64))]
    dataloader = InMemoryDataLoader(dataset=pytree, minibatch_size=4)
    assert dataloader.minibatch_count() == 8
    for x in dataloader.minibatches(None):
        assert x[0].shape == (4, 64, 64), x[0].shape


if __name__ == "__main__":
    _check()
