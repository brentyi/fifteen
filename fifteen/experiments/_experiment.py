"""Simple experiment manager for managing metadata, logs, and checkpoints.

Source: https://github.com/brentyi/dfgo/blob/master/lib/experiment_files.py
"""

import dataclasses
import pathlib
import shutil
from typing import Any, Dict, Optional, Type, TypeVar, Union, overload

import flax.metrics.tensorboard
import flax.training.checkpoints
import jax_dataclasses
import numpy as onp
import yaml
from jax import numpy as jnp

T = TypeVar("T")
PytreeType = TypeVar("PytreeType")
Pytree = Any

Array = Union[jnp.ndarray, onp.ndarray]
ArrayOrFloat = Union[Array, float]


@jax_dataclasses.pytree_dataclass
class TensorboardLogData:
    scalars: Dict[str, ArrayOrFloat] = jax_dataclasses.field(default_factory=dict)
    histograms: Dict[str, Array] = jax_dataclasses.field(default_factory=dict)

    def prefix(self, prefix: str) -> "TensorboardLogData":
        """Add a prefix to all contained tag names. Useful for scoping, or creating those little
        folders in the Tensorboard scalar view."""

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
        scalars: Dict[str, ArrayOrFloat] = {},
    ) -> "TensorboardLogData":
        return self.merge(
            TensorboardLogData(scalars=scalars, histograms={}),
        )

    def merge_histograms(
        self,
        histograms: Dict[str, Array] = {},
    ) -> "TensorboardLogData":
        return self.merge(
            TensorboardLogData(scalars={}, histograms=histograms),
        )


@dataclasses.dataclass(frozen=True)
class Experiment:
    """Helper class for locating checkpoints, logs, and any experiment metadata."""

    identifier: str
    verbose: bool = True

    # Generated in __post_init__
    data_dir: pathlib.Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # Assign checkpoint + log directories
        root = pathlib.Path("./experiments")
        super().__setattr__("data_dir", root / self.identifier)

    def assert_new(self) -> "Experiment":
        """Makes sure that there are no existing checkpoints, logs, or metadata. Returns
        self."""
        assert not self.data_dir.exists() or tuple(self.data_dir.iterdir()) == ()
        return self

    def assert_exists(self) -> "Experiment":
        """Makes sure that there are existing checkpoints, logs, or metadata. Returns
        self."""
        assert self.data_dir.exists() and tuple(self.data_dir.iterdir()) != ()
        return self

    def clear(self) -> "Experiment":
        """Delete all checkpoints, logs, and metadata associated with an experiment.
        Returns self."""

        def error_cb(func: Any, path: str, exc_info: Any) -> None:
            """Error callback for shutil.rmtree."""
            self._print(f"Error deleting {path}")

        def delete_path(path: pathlib.Path, n: int = 5) -> None:
            """Deletes a path, as well as up to `n` empty parent directories."""
            if not path.exists():
                return

            shutil.rmtree(path, onerror=error_cb)
            self._print("Deleting", path)

            if n > 0 and len(list(path.parent.iterdir())) == 0:
                delete_path(path.parent, n - 1)

        delete_path(self.data_dir)

        return self

    def move(self, new_identifier: str) -> "Experiment":
        """Move all files corresponding to an experiment to a new identifier. Returns
        updated Experiment object."""
        new_experiment = Experiment(identifier=new_identifier, verbose=self.verbose)

        def move(src: pathlib.Path, dst=pathlib.Path) -> None:
            if not src.exists():
                return
            self._print("Moving {src} to {dst}")
            shutil.move(src=str(src), dst=str(dst))

        move(src=self.data_dir, dst=new_experiment.data_dir)

        return new_experiment

    def write_metadata(self, name: str, object: Any) -> None:
        """Serialize an object as a yaml file, then save it to the experiment's metadata
        directory."""
        self._ensure_directory_exists(self.data_dir)
        assert not name.endswith(".yaml")

        path = self.data_dir / (name + ".yaml")
        self._print("Writing metadata to", path)
        with open(path, "w") as file:
            file.write(yaml.dump(object))

    def read_metadata(self, name: str, expected_type: Type[T]) -> T:
        """Load an object from the experiment's metadata directory."""
        path = self.data_dir / (name + ".yaml")

        self._print("Reading metadata from", path)
        with open(path, "r") as file:
            output = yaml.load(
                file.read(),
                Loader=yaml.Loader,  # Unsafe loading!
            )
        assert isinstance(output, expected_type)
        return output

    def save_checkpoint(
        self,
        target: Pytree,
        step: int,
        prefix: str = "checkpoint_",
        keep: int = 1,
    ) -> str:
        """Thin wrapper around flax's `save_checkpoint()` function.
        Returns a file name, as a string."""
        self._ensure_directory_exists(self.data_dir)
        filename = flax.training.checkpoints.save_checkpoint(
            ckpt_dir=self.data_dir,
            target=target,
            step=step,
            prefix=prefix,
            keep=keep,
        )
        self._print("Saved checkpoint to", filename)
        return filename

    def restore_checkpoint(
        self,
        target: PytreeType,
        step: Optional[int] = None,
        prefix: str = "checkpoint_",
    ) -> PytreeType:
        """Thin wrapper around flax's `restore_checkpoint()` function."""
        state_dict = flax.training.checkpoints.restore_checkpoint(
            ckpt_dir=self.data_dir,
            target=None,  # Allows us to assert that a checkpoint was actually found
            step=step,
            prefix=prefix,
        )
        if state_dict is None:
            raise FileNotFoundError("No checkpoint found!")
        self._print(f"Successfully loaded checkpoint!")
        return flax.serialization.from_state_dict(target, state_dict)

    @property
    def summary_writer(self) -> flax.metrics.tensorboard.SummaryWriter:
        """Helper for Tensorboard logging."""
        if not hasattr(self, "__summary_writer__"):
            object.__setattr__(
                self,
                "__summary_writer__",
                flax.metrics.tensorboard.SummaryWriter(log_dir=self.data_dir),
            )
        return object.__getattribute__(self, "__summary_writer__")

    def log(
        self,
        log_data: TensorboardLogData,
        step: int,
        log_scalars_every_n: int = 1,
        log_histograms_every_n: int = 1,
    ):
        """Logging helper for Tensorboard."""
        # In the future, we could make this JIT-friendly with a host callback.
        if step % log_scalars_every_n == 0:
            for k, v in log_data.scalars.items():
                self.summary_writer.scalar(k, v, step=step)
        if step % log_histograms_every_n == 0:
            for k, v in log_data.histograms.items():
                self.summary_writer.histogram(k, v, step=step)

    def _ensure_directory_exists(self, path: pathlib.Path) -> None:
        """Helper for... ensuring that directories exist."""
        if not path.exists():
            path.mkdir(parents=True)
            self._print(f"Made directory at {path}")

    def _print(self, *args, **kwargs) -> None:
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        if self.verbose:
            print(f"[{type(self).__name__}-{self.identifier}]", *args, **kwargs)
