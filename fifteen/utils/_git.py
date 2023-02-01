import pathlib
import subprocess
from typing import Optional


def get_git_commit_hash(cwd: Optional[pathlib.Path] = None) -> str:
    """Returns the current Git commit hash."""
    if cwd is None:
        cwd = pathlib.Path.cwd()
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd.as_posix())
        .decode("ascii")
        .strip()
    )


def get_git_diff(cwd: Optional[pathlib.Path] = None) -> str:
    """Returns the output of `git diff HEAD`."""
    if cwd is None:
        cwd = pathlib.Path.cwd()
    return (
        subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd.as_posix())
        .decode("ascii")
        .strip()
    )
