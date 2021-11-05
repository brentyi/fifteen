from typing import Any, Callable

from jax.experimental import host_callback as hcb


def _hcb_print(
    self,
    string_from_args: Callable[..., str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Helper for printing via a host callback. JIT-friendly!"""
    hcb.id_tap(
        lambda args_kwargs, _unused_transforms: print(
            string_from_args(*args_kwargs[0], **args_kwargs[1]),
        ),
        (args, kwargs),
    )
