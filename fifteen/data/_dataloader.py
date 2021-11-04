"""Generic dataloader implementation, along with an associated dataset specification.

Similar to the Pytorch dataloader, but stateless."""

import dataclasses
import math
from multiprocess import Manager, Pool, Queue
from typing import Callable, Generic, Iterable, List, Optional, Protocol, TypeVar

import jax
import numpy as onp

T = TypeVar("T")
CollateFunction = Callable[[List[T]], T]


DatasetT = TypeVar("DatasetT", covariant=True)
DataLoaderT = TypeVar("DataLoaderT")


class DatasetProtocol(Protocol, Generic[DatasetT]):
    """Protocol for defining datasets."""

    def __getitem__(self, index: int) -> DatasetT:
        ...

    def __len__(self) -> int:
        ...


PytreeType = TypeVar("PytreeType")


@dataclasses.dataclass(frozen=True)
class _MultiprocessFields:
    """Attributes needed for multiprocessing dataloaders."""

    num_workers: int
    pool: Pool  # We're currently not closing these correctly.
    result_queue: Queue


@dataclasses.dataclass(frozen=True)
class DataLoader(Generic[DataLoaderT]):
    """Dataloader. Inspired by Pytorch's, but stateless.

    Expects an arbitrary indexable dataset, which should map integer indices to items as
    arrays or Pytrees. `.minibatches()` can then be used to construct an (optionally
    shuffled) iterable over minibatches of stacked items."""

    # Only two required arguments: our dataset and batch size.
    dataset: DatasetProtocol[DataLoaderT]
    batch_size: int

    num_workers: int = 0
    """Set to 0 to disable multiprocessing."""

    drop_last: bool = True
    """Drop last minibatch if dataset is not evenly divisible."""

    collate_fn: CollateFunction = lambda items: jax.tree_map(
        lambda *arrays: onp.stack(arrays, axis=0), *items
    )
    """Collate function. By default, we simply stack along `axis=0`."""

    multiprocess_fields: Optional[_MultiprocessFields] = dataclasses.field(init=False)

    def __post_init__(self):
        # Create a process pool on instantiation, if `num_workers > 0`.
        object.__setattr__(
            self,
            "multiprocess_fields",
            _MultiprocessFields(
                num_workers=self.num_workers,
                pool=Pool(processes=self.num_workers),
                result_queue=Manager().Queue(maxsize=self.num_workers * 2),
            )
            if self.num_workers > 0
            else None,
        )

    def minibatches(self, seed: Optional[int]) -> Iterable[DataLoaderT]:
        """Returns an iterable over minibatches for our dataset. Optionally shuffled using
        a random seed."""

        indices = onp.arange(len(self.dataset))
        if seed is not None:
            onp.random.default_rng(seed=seed).shuffle(indices)

        minibatch_count = indices.shape[0] / self.batch_size
        if self.drop_last:
            indices = indices[: math.floor(minibatch_count) * self.batch_size]
            minibatch_count = math.floor(minibatch_count)
        else:
            minibatch_count = math.ceil(minibatch_count)

        return _Minibatches(
            dataloader=self,
            indices=indices,
            minibatch_count=minibatch_count,
        )


@dataclasses.dataclass
class _Minibatches(Iterable[DataLoaderT], Generic[DataLoaderT]):
    """Iterable object for returning minibatches, with async support."""

    dataloader: DataLoader[DataLoaderT]
    indices: onp.ndarray  # Shape: (dataset length,)
    minibatch_count: int

    def __iter__(self):
        mp_fields = self.dataloader.multiprocess_fields
        collate_fn = self.dataloader.collate_fn
        if mp_fields is None:
            # Simple synchronous iterator.
            for i in range(self.minibatch_count):
                indices = self._get_minibatch_indices(i)
                items = [self.dataloader.dataset[i] for i in indices]
                yield collate_fn(items)
        else:
            dataset = self.dataloader.dataset
            result_queue = mp_fields.result_queue
            mp_fields.pool.map_async(
                lambda indices: result_queue.put(
                    collate_fn(list(map(dataset.__getitem__, indices)))
                ),
                [self._get_minibatch_indices(i) for i in range(self.minibatch_count)],
                error_callback=print,
            )
            for i in range(self.minibatch_count):
                yield result_queue.get()

    def _get_minibatch_indices(self, index: int) -> onp.ndarray:
        start_index = self.dataloader.batch_size * index
        end_index = min(
            self.dataloader.batch_size * (index + 1), len(self.dataloader.dataset)
        )
        return self.indices[start_index:end_index]

    def __len__(self):
        return self.minibatch_count


def main() -> None:
    import time

    from tqdm.auto import tqdm

    def benchmark(dataset: DatasetProtocol[onp.ndarray]):
        """Benchmark some simulated training job for both a synchronous and an async dataloader."""
        loaders = {
            "synchronous": DataLoader(dataset, batch_size=64, num_workers=0),
            "async": DataLoader(dataset, batch_size=64, num_workers=4),
        }

        for name, loader in loaders.items():
            start_time = time.time()
            for minibatch in tqdm(loader.minibatches(seed=0)):
                time.sleep(0.1)
            print(
                f"Simulated training time on {name} dataloader:",
                time.time() - start_time,
            )

    @dataclasses.dataclass
    class DummyDataset:
        delay: float

        def __getitem__(self, index):
            time.sleep(self.delay)
            return onp.random.randn(64, 64, 3)

        def __len__(self):
            return 500

    print("Simple dataset")
    benchmark(DummyDataset(delay=0.0))
    print()
    print("IO bound dataset")
    benchmark(DummyDataset(delay=0.01))


if __name__ == "__main__":
    main()
