import collections
import dataclasses
from typing import Iterable, Optional, TypeVar, Union, cast, overload

import jax

from ._protocols import SizedIterable

PyTreeType = TypeVar("PyTreeType")


@overload
def prefetching_map(
    inputs: SizedIterable[PyTreeType],
    device: Optional[jax.Device] = None,
    buffer_size: int = 2,
) -> SizedIterable[PyTreeType]: ...


@overload
def prefetching_map(
    inputs: Iterable[PyTreeType],
    device: Optional[jax.Device] = None,
    buffer_size: int = 2,
) -> Iterable[PyTreeType]: ...


def prefetching_map(
    inputs: Union[Iterable[PyTreeType], SizedIterable[PyTreeType]],
    device: Optional[jax.Device] = None,
    buffer_size: int = 2,
) -> Union[Iterable[PyTreeType], SizedIterable[PyTreeType]]:
    """Maps iterables over PyTrees to an identical iterable, but with a prefetching
    buffer under the hood. Adapted from `flax.jax_utils.prefetch_to_device()`.

    This can improve parallelization for GPUs, particularly when memory is re-allocated
    before freeing is finished. When the buffer size is set to 2, we make it explicit
    that two sets of data should live in GPU memory at once: for a standard training
    loop, this is typically both the "current" minibatch and the "next" one.

    If a device is specified, we commit arrays (via `jax.device_put()`) before pushing them
    onto the buffer. This should generally be set if the input iterable yields arrays
    that are still living on the CPU.

    For multi-device use cases, we can combine this function with
    :meth:`fifteen.data.sharding_map()`."""

    if hasattr(inputs, "__len__"):
        return _PrefetchingMapSized(
            cast(SizedIterable[PyTreeType], inputs), device, buffer_size
        )
    else:
        return _PrefetchingMap(inputs, device, buffer_size)


@dataclasses.dataclass
class _PrefetchingMap(Iterable[PyTreeType]):
    inputs: Iterable[PyTreeType]
    device: Optional[jax.Device]
    buffer_size: int

    def __iter__(self):
        """Adapted from `flax.jax_utils.prefetch_to_device()`:
        https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        """
        queue = collections.deque()
        input_iter = iter(self.inputs)

        def try_enqueue() -> None:
            try:
                item = next(input_iter)
            except StopIteration:
                return

            if self.device is not None:
                item = jax.device_put(item, device=self.device)
            queue.append(item)

        for i in range(self.buffer_size):
            try_enqueue()
        while len(queue) > 0:
            yield queue.pop()
            try_enqueue()


@dataclasses.dataclass
class _PrefetchingMapSized(_PrefetchingMap[PyTreeType], SizedIterable[PyTreeType]):
    inputs: SizedIterable[PyTreeType]

    def __len__(self) -> int:
        return len(self.inputs)
