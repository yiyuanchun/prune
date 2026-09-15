"""Scope optional diagnostic I/O without changing verification computations."""
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
import os
from pathlib import Path
from tempfile import TemporaryDirectory


_save_artifacts = ContextVar('save_verification_artifacts', default=True)


def artifacts_enabled():
    return _save_artifacts.get()


@contextmanager
def diagnostic_artifacts(enabled):
    token = _save_artifacts.set(bool(enabled))
    try:
        yield
    finally:
        _save_artifacts.reset(token)


@contextmanager
def sample_workspace(output_path):
    if artifacts_enabled():
        output_path.mkdir()
        yield output_path
    else:
        # REDNet must still serialize/reload its network for equivalence checks.
        # Only this required transient artifact is written in lightweight mode.
        with TemporaryDirectory(prefix='parnv-mnist-') as directory:
            yield Path(directory)


@contextmanager
def solver_console():
    if artifacts_enabled():
        yield
    else:
        # Preserve stderr/tracebacks; discard verbose Python solver progress.
        # Marabou already runs at verbosity=0. No in-memory log buffer is kept.
        with open(os.devnull, 'w') as stream, redirect_stdout(stream):
            yield
