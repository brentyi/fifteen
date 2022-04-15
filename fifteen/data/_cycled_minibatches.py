from typing import Iterator, Optional, TypeVar

from ._protocols import DataLoaderProtocol

PyTreeType = TypeVar("PyTreeType")


def cycled_minibatches(
    dataloader: DataLoaderProtocol[PyTreeType],
    shuffle_seed: Optional[int],
) -> Iterator[PyTreeType]:
    """Iterate over items in a dataloader infinitely. Abstracts away the concept of
    'epochs', which are often logical for dataloaders to reason about, but can be a
    misleading metric for training progress, particularly when datasets of different
    sizes or data augmentation is involved.

    Under the hood, shuffling is handled by incrementing the shuffle seed by 1 for each
    (implicit) epoch."""

    while True:
        for minibatch in dataloader.minibatches(shuffle_seed):
            yield minibatch

        if shuffle_seed is not None:
            shuffle_seed += 1
