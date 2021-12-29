"""Structural supertypes for working with datasets and dataloaders."""

from typing import Iterable, Optional, Protocol, Sized, TypeVar

# TypeVar for defining protocols; note that method types can be narrowed and thus this
# must be covariant.
ContainedType = TypeVar("ContainedType", covariant=True)


class SizedIterable(Iterable[ContainedType], Sized, Protocol[ContainedType]):
    """Protocol for objects that define both `__iter__` and `__len__` methods.

    This is particularly useful for managing minibatches, which can be iterated over but
    only in order (due to multiprocessing/prefetching optimizations), and for which
    length evaluation is useful for tools like `tqdm`."""


class MapDatasetProtocol(Protocol[ContainedType]):
    """Protocol for defining PyTorch-style "map" datasets, which implement two methods:
    __getitem__() for loading single samples and __len__() for counting the total
    number of samples.

    This is similar to collections.abc.Mapping, but does not require implementations of
    __contains__."""

    def __getitem__(self, index: int) -> ContainedType:
        ...

    def __len__(self) -> int:
        ...


class DataLoaderProtocol(Protocol[ContainedType]):
    """Protocol for dataloaders, which are used to generate minibatches that can be
    iterated over."""

    minibatch_size: int
    drop_last: bool

    def minibatch_count(self) -> int:
        ...

    def minibatches(self, shuffle_seed: Optional[int]) -> SizedIterable[ContainedType]:
        ...
