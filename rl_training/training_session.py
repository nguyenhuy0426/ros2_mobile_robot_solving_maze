"""Exclusive ownership of the shared simulator for a training session."""

from contextlib import contextmanager
import fcntl
import os


@contextmanager
def training_lock(path):
    # Never unlink the file: another process may already have its inode open.
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another multi-robot trainer owns the simulator ({path}). "
                "Use one trainer for the shared robot topics.") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
