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
    """Data structure for managing data meant for logging to Tensorboard."""

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
