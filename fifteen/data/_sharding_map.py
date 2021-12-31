import dataclasses
from typing import Iterable, Sequence, TypeVar, Union, cast, overload

import jax
from jax import numpy as jnp
from jax.lib import xla_client

from ._protocols import SizedIterable

PyTreeType = TypeVar("PyTreeType")


@overload
def sharding_map(
    inputs: SizedIterable[PyTreeType],
    devices: Sequence[xla_client.Device],
) -> SizedIterable[PyTreeType]:
    ...


@overload
def sharding_map(
    inputs: Iterable[PyTreeType],
    devices: Sequence[xla_client.Device],
) -> Iterable[PyTreeType]:
    ...


def sharding_map(
    inputs: Union[Iterable[PyTreeType], SizedIterable[PyTreeType]],
    devices: Sequence[xla_client.Device],
) -> Union[Iterable[PyTreeType], SizedIterable[PyTreeType]]:
    """Maps iterables over PyTrees to iterables over sharded PyTrees, which are
    distributed on multiple devices.

    Takes as input leaf shapes
        (N, ...)
    and maps as output to an iterable with leaf shapes
        (device_count, N // device_count, ...).
    where the leading axis corresponds to the index of the device that each shard is
    committed to."""

    if hasattr(inputs, "__len__"):
        return _ShardingMapSized(cast(SizedIterable[PyTreeType], inputs), devices)
    else:
        return _ShardingMap(inputs, devices)


@dataclasses.dataclass
class _ShardingMap(Iterable[PyTreeType]):
    inputs: Iterable[PyTreeType]
    devices: Sequence[xla_client.Device]

    def __iter__(self):
        device_count = len(self.devices)

        def shard(leaf: jnp.ndarray) -> jnp.ndarray:
            assert (
                leaf.shape[0] % device_count == 0
            ), "Batch size must be divisible by device count."

            leaf = leaf.reshape(
                (device_count, leaf.shape[0] // device_count) + leaf.shape[1:]
            )
            return jax.device_put_sharded(list(leaf), devices=self.devices)

        return iter(map(lambda pytree: jax.tree_map(shard, pytree), self.inputs))


@dataclasses.dataclass
class _ShardingMapSized(_ShardingMap[PyTreeType], SizedIterable[PyTreeType]):
    inputs: SizedIterable[PyTreeType]

    def __len__(self) -> int:
        return len(self.inputs)
