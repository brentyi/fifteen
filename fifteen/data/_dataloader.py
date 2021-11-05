"""Generic dataloader implementation, along with an associated dataset specification.

Similar to the Pytorch dataloader, but stateless."""

import dataclasses
import math
from typing import Callable, Dict, Generic, Iterable, List, Optional, TypeVar

import jax
import numpy as onp
from multiprocess import Process, Queue
from typing_extensions import Protocol

T = TypeVar("T")
CollateFunction = Callable[[List[T]], T]


DatasetT = TypeVar("DatasetT", covariant=True)
DataLoaderT = TypeVar("DataLoaderT")


class DatasetProtocol(Protocol[DatasetT]):
    """Protocol for defining datasets."""

    def __getitem__(self, index: int) -> DatasetT:
        ...

    def __len__(self) -> int:
        ...


PytreeType = TypeVar("PytreeType")


def _worker_loop(
    dataset: DatasetProtocol,
    index_queue: Queue,
    result_queue: Queue,
    collate_fn: CollateFunction,
) -> None:
    """Worker for dataloaders with multiprocessing."""
    while True:
        i, indices = index_queue.get()
        result_queue.put((i, collate_fn([dataset[i] for i in indices])))


@dataclasses.dataclass(frozen=True)
class _WorkersState:
    """Objects needed for managing and cleaning up after workers.."""

    workers: List[Process]
    index_queue: Queue
    result_queue: Queue

    def __del__(self):
        """Clean up workers."""
        self.index_queue.close()
        self.result_queue.close()
        for w in self.workers:
            w.kill()
            w.join()
            w.close()


@dataclasses.dataclass(frozen=True)
class DataLoader(Generic[DataLoaderT]):
    """Dataloader. Inspired by Pytorch's, but stateless.

    Expects an arbitrary indexable dataset, which should map integer indices to items as
    arrays or Pytrees. `.minibatches()` can then be used to construct an (optionally
    shuffled) iterable over minibatches of stacked items."""

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

    workers_state: Optional[_WorkersState] = dataclasses.field(init=False)

    def __post_init__(self):
        # Create workers on instantiation.
        if self.num_workers == 0:
            object.__setattr__(self, "workers_state", None)
        else:
            # m = Manager()
            index_queue = Queue()
            result_queue = Queue(maxsize=self.num_workers)

            assert self.num_workers > 0
            workers = []
            for i in range(self.num_workers):
                w = Process(
                    target=_worker_loop,
                    args=(
                        self.dataset,
                        index_queue,
                        result_queue,
                        self.collate_fn,
                    ),
                )
                w.daemon = True
                w.start()
                workers.append(w)

            object.__setattr__(
                self,
                "workers_state",
                _WorkersState(
                    workers=workers,
                    index_queue=index_queue,
                    result_queue=result_queue,
                ),
            )

    def minibatches(self, shuffle_seed: Optional[int]) -> Iterable[DataLoaderT]:
        """Returns an iterable over minibatches for our dataset. Optionally shuffled using
        a random seed."""

        indices = onp.arange(len(self.dataset))
        if shuffle_seed is not None:
            onp.random.default_rng(seed=shuffle_seed).shuffle(indices)

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
        dataset = self.dataloader.dataset
        collate_fn = self.dataloader.collate_fn
        num_workers = self.dataloader.num_workers
        if num_workers == 0:
            # Simple synchronous iterator.
            for i in range(self.minibatch_count):
                indices = self._get_minibatch_indices(i)
                items = [dataset[i] for i in indices]
                yield collate_fn(items)
        else:
            mp_fields = self.dataloader.workers_state
            result_queue = mp_fields.result_queue
            queued_minibatches = 0

            # Immediately put all minibatch indices on the index queue. This might be problematic
            # for extremely large datasets.
            for i in range(self.minibatch_count):
                mp_fields.index_queue.put(
                    (i, self._get_minibatch_indices(queued_minibatches))
                )

            # Yield minibatches in ascending order; note that they may be shuffled when
            # coming off of the queue.
            minibatch_cache: Dict[int, DataLoaderT] = {}
            for i in range(self.minibatch_count):
                if i not in minibatch_cache:
                    while True:
                        received_i, minibatch = result_queue.get()
                        if received_i == i:
                            yield minibatch
                            break
                        else:
                            minibatch_cache[received_i] = minibatch
                else:
                    yield minibatch_cache.pop(i)

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
            for minibatch in tqdm(loader.minibatches(shuffle_seed=0)):
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
