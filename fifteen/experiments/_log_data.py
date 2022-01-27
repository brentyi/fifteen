from typing import Dict, TypeVar, Union

import jax
import jax_dataclasses
import numpy as onp
from jax import numpy as jnp

T = TypeVar("T")
Array = Union[jnp.ndarray, onp.ndarray]
Scalar = Union[Array, float, int]


@jax_dataclasses.pytree_dataclass
class TensorboardLogData:
    """Data structure for logging to Tensorboard."""

    scalars: Dict[str, Scalar] = jax_dataclasses.field(default_factory=dict)
    histograms: Dict[str, Array] = jax_dataclasses.field(default_factory=dict)

    def prefix(self, prefix: str) -> "TensorboardLogData":
        """Add a prefix to all contained tag names. Useful for scoping, or creating
        those folders in the Tensorboard scalar view."""

        def _prefix(x: Dict[str, T]) -> Dict[str, T]:
            return {prefix + k: v for k, v in x.items()}

        return TensorboardLogData(
            scalars=_prefix(self.scalars),
            histograms=_prefix(self.histograms),
        )

    def merge(self, other: "TensorboardLogData") -> "TensorboardLogData":
        """Merge two log data structures."""
        return TensorboardLogData(
            scalars=dict(**self.scalars, **other.scalars),
            histograms=dict(**self.histograms, **other.histograms),
        )

    def merge_scalars(
        self,
        scalars: Dict[str, Scalar] = {},
    ) -> "TensorboardLogData":
        return TensorboardLogData(
            scalars=dict(**self.scalars, **scalars),
            histograms=self.histograms,
        )

    def merge_histograms(
        self,
        histograms: Dict[str, Array] = {},
    ) -> "TensorboardLogData":
        return TensorboardLogData(
            scalars=self.scalars,
            histograms=dict(**self.histograms, **histograms),
        )

    def fix_sharded_scalars(self) -> "TensorboardLogData":
        """When log data is returned from a function transformed by `pmap`, scalars will
        often be returned as sharded arrays, distributed across multiple devices. This
        makes them no longer scalars, and breaks compatibility with standard logging
        utilities.

        To fix this, we replace each sharded array in the scalar dictionary with the
        first value from the flattened representation. Histogram data is unmodified.

        In the future, this might also support averaging across the scalars, but since
        an arithmetic mean doesn't make sense for many metrics the current approach is
        to simply call `jax.lax.pmean` in the pmapped function. Some performance
        analysis could be done here."""

        scalars: Dict[str, Scalar] = {}
        for k, v in self.scalars.items():
            assert isinstance(v, jax.lib.xla_extension.pmap_lib.ShardedDeviceArray)

            # We pull out the first value on the first device.
            scalars[k] = v[tuple(0 for _ in range(len(v.shape)))]

        return TensorboardLogData(scalars=scalars, histograms=self.histograms)
