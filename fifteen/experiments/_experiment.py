"""Simple experiment manager for managing metadata, logs, and checkpoints.

Source: https://github.com/brentyi/dfgo/blob/master/lib/experiment_files.py
"""
import dataclasses
import functools
import pathlib
import shutil
from typing import Any, Optional, Type, TypeVar

import dcargs
import flax.metrics.tensorboard
import flax.training.checkpoints
import jax
import yaml
from typing_extensions import get_origin

from ._log_data import TensorboardLogData

T = TypeVar("T")
PytreeType = TypeVar("PytreeType")
Pytree = Any


try:
    # Python >=3.8.
    cached_property = functools.cached_property
except AttributeError:
    # Python <=3.7.
    # Given the usage context it's very unlikely to matter practically, but note that
    # this could lead to memory leaks.
    def cached_property(method):  # type: ignore
        return property(functools.lru_cache(maxsize=None)(method))


def _get_origin(cls: Type) -> Type:
    """Get origin type; helpful for unwrapping generics, etc."""
    origin = get_origin(cls)
    return cls if origin is None else origin


@dataclasses.dataclass(frozen=True)
class Experiment:
    """We define an "experiment" as a simple directory, where files associated with some
    run of a training script are co-located.

    There's very little real code here; instead we use a common experiment data
    directory to implement thin wrappers around:
    - `flax.training.checkpoints` for checkpointing.
    - `PyYAML` and `dcargs` for serializing metadata.
    - `flax.metrics.tensorboard.SummaryWriter` for logging.
    """

    data_dir: pathlib.Path
    verbose: bool = True

    #  YAML-based metadata utilities.

    def write_metadata(self, name: str, object: Any) -> None:
        """Serialize an object as a yaml file, then save it to the experiment's metadata
        directory. Includes special handling for dataclasses (via dcargs)."""
        self._mkdir_if_needed(self.data_dir)
        assert not name.endswith(".yaml")

        path = self.data_dir / (name + ".yaml")
        self._print("Writing metadata to", path)
        with open(path, "w") as file:
            file.write(
                dcargs.to_yaml(object)
                if dataclasses.is_dataclass(object)
                else yaml.dump(object)
            )

    def read_metadata(self, name: str, expected_type: Type[T]) -> T:
        """Load an object from the experiment's metadata directory. Includes special
        handling for dataclasses (via dcargs)."""
        path = self.data_dir / (name + ".yaml")

        self._print("Reading metadata from", path)
        with open(path, "r") as file:
            output = (
                dcargs.from_yaml(expected_type, file.read())
                if dataclasses.is_dataclass(_get_origin(expected_type))
                else yaml.load(
                    file.read(),
                    Loader=yaml.Loader,  # Unsafe loading!
                )
            )
        assert isinstance(output, expected_type)
        return output

    #  Checkpointing helpers.

    def save_checkpoint(
        self,
        target: Pytree,
        step: int,
        prefix: str = "checkpoint_",
        keep: int = 1,
        overwrite: bool = False,
        keep_every_n_steps: Optional[int] = None,
    ) -> str:
        """Thin wrapper around flax's `save_checkpoint()` function.
        Returns a file name, as a string."""
        self._mkdir_if_needed(self.data_dir)
        filename = flax.training.checkpoints.save_checkpoint(
            ckpt_dir=str(self.data_dir),
            target=target,
            step=step,
            prefix=prefix,
            keep=keep,
            overwrite=overwrite,
            keep_every_n_steps=keep_every_n_steps,
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
            ckpt_dir=str(self.data_dir),
            target=None,  # Allows us to raise an error if no checkpoint was found.
            step=step,
            prefix=prefix,
        )
        if state_dict is None:
            raise FileNotFoundError("No checkpoint found!")
        self._print("Successfully loaded checkpoint!")
        return flax.serialization.from_state_dict(target, state_dict)

    #  Tensorboard logging helpers.

    @cached_property
    def summary_writer(self) -> flax.metrics.tensorboard.SummaryWriter:
        """Property for accessing a summary writer for Tensorboard logging."""
        return flax.metrics.tensorboard.SummaryWriter(log_dir=str(self.data_dir))

    def log(
        self,
        log_data: TensorboardLogData,
        step: int,
        log_scalars_every_n: int = 1,
        log_histograms_every_n: int = 1,
    ):
        """Logging helper for Tensorboard.

        For TensorboardLogData instances returned from `pmap`-transformed functions, see
        `TensorboardLogData.fix_sharded_scalars()`."""
        # We could easily make this JIT-friendly with a host callback, but that might
        # encourage unnecessarily un-modular design patterns.
        if step % log_scalars_every_n == 0:
            for k, v in log_data.scalars.items():
                if hasattr(v, "shape"):
                    shape = v.shape  # type: ignore
                    assert shape == (), (
                        f"Got {shape=} instead of a scalar. For use with `jax.pmap`,"
                        "`log_data.fix_sharded_scalars()` may be helpful."
                    )
                self.summary_writer.scalar(k, v, step=step)
        if step % log_histograms_every_n == 0:
            for k, v in log_data.histograms.items():
                self.summary_writer.histogram
                self.summary_writer.histogram(k, v, step=step)

    #  Helpers for common "experiment management" operations.

    def assert_new(self) -> "Experiment":
        """Makes sure that there are no existing checkpoints, logs, or metadata. Returns
        self."""

        assert not self.data_dir.exists() or next(self.data_dir.iterdir(), None) is None
        return self

    def assert_exists(self) -> "Experiment":
        """Makes sure that there are existing checkpoints, logs, or metadata. Returns
        self."""
        assert (
            self.data_dir.exists() and next(self.data_dir.iterdir(), None) is not None
        )
        return self

    def clear(self) -> "Experiment":
        """Deletes `self.data_dir`. This clears all checkpoints, logs, and metadata
        inside of it. Returns self."""

        def error_cb(func: Any, path: str, exc_info: Any) -> None:
            """Error callback for shutil.rmtree."""
            self._print(f"Error deleting {path}")

        def rm_recursive(path: pathlib.Path, n: int = 5) -> None:
            """Deletes a path, as well as up to `n` empty parent directories."""
            if not path.exists():
                return

            shutil.rmtree(path, onerror=error_cb)
            self._print("Deleting", path)

            if n > 0 and len(list(path.parent.iterdir())) == 0:
                rm_recursive(path.parent, n - 1)

        rm_recursive(self.data_dir)

        return self

    def move(self, new_data_dir: pathlib.Path) -> "Experiment":
        """Move all files corresponding to an experiment to a new location. Returns
        updated Experiment object."""
        new_experiment = Experiment(data_dir=new_data_dir, verbose=self.verbose)

        def move(src: pathlib.Path, dst=pathlib.Path) -> None:
            if not src.exists():
                return
            self._print("Moving {src} to {dst}")
            shutil.move(src=str(src), dst=str(dst))

        move(src=self.data_dir, dst=new_experiment.data_dir)

        return new_experiment

    # Private helpers.

    def _mkdir_if_needed(self, path: pathlib.Path) -> None:
        if not path.exists():
            path.mkdir(parents=True)
            self._print(f"Made directory at {path}")

    def _print(self, *args, **kwargs) -> None:
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        # TODO: re-consider the overhead/flexibility tradeoff of a proper logging setup.
        if self.verbose:
            print(f"[{type(self).__name__}]", *args, **kwargs)
