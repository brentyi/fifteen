import dataclasses
import time
from typing import Generator, Iterable, Tuple, overload

from ..data import SizedIterable


@dataclasses.dataclass
class LoopMetrics:
    counter: int
    iterations_per_sec: float
    time_elapsed: float


@overload
def range_with_metrics(stop: int, /) -> SizedIterable[LoopMetrics]:
    ...


@overload
def range_with_metrics(start: int, stop: int, /) -> SizedIterable[LoopMetrics]:
    ...


@overload
def range_with_metrics(
    start: int, stop: int, step: int, /
) -> SizedIterable[LoopMetrics]:
    ...


def range_with_metrics(*args: int) -> SizedIterable[LoopMetrics]:
    """Light wrapper for `fifteen.utils.loop_metric_generator()`, for use in place of
    `range()`. Yields a LoopMetrics object instead of an integer."""
    return _RangeWithMetrics(args=args)


@dataclasses.dataclass
class _RangeWithMetrics:
    args: Tuple[int, ...]

    def __iter__(self):
        loop_metrics = loop_metric_generator()
        for counter in range(*self.args):
            yield dataclasses.replace(next(loop_metrics), counter=counter)

    def __len__(self) -> int:
        return len(range(*self.args))


def loop_metric_generator() -> Generator[LoopMetrics, None, None]:
    """Generator for computing loop metrics.

    Note that the first `iteration_per_sec` metric will be 0.0.

    Example usage:
    ```
    # Note that this is an infinite loop.
    for metric in loop_metric_generator():
        time.sleep(1.0)
        print(metric)
    ```

    or:
    ```
    loop_metrics = loop_metric_generator()
    while True:
        time.sleep(1.0)
        print(next(loop_metrics).iterations_per_sec)
    ```
    """

    counter = 0
    time_start = time.time()
    time_prev = time_start
    while True:
        time_now = time.time()
        yield LoopMetrics(
            counter=counter,
            iterations_per_sec=1.0 / (time_now - time_prev) if counter > 0 else 0.0,
            time_elapsed=time_now - time_start,
        )
        time_prev = time_now
        counter += 1


if __name__ == "__main__":
    for loop_metrics in range_with_metrics(10):
        time.sleep(1.0)
        print(loop_metrics)

    generator = loop_metric_generator()
    for metrics in loop_metric_generator():
        time.sleep(1.0)
        print(metrics)
        print(next(generator))
