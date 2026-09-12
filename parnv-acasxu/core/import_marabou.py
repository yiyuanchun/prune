import os
import sys

from core.configuration.consts import CODE_DIR, VERBOSE


def _candidate_marabou_dirs(explicit_marabou_dir=None):
    candidates = []
    for value in (
            explicit_marabou_dir,
            os.environ.get("NARV_MARABOU_DIR"),
            os.environ.get("MARABOU_DIR"),
            os.path.join(CODE_DIR, "Marabou"),
    ):
        if not value:
            continue
        candidate = os.path.abspath(os.fspath(value))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def dynamically_import_marabou(query_type="adversarial", marabou_dir=None):
    if query_type not in {"adversarial", "basic", "acas_xu_conjunction"}:
        raise KeyError("Unsupported Marabou query type '{}'.".format(query_type))

    tried_dirs = _candidate_marabou_dirs(explicit_marabou_dir=marabou_dir)
    for candidate in tried_dirs:
        maraboupy_dir = os.path.join(candidate, "maraboupy")
        if candidate not in sys.path:
            sys.path.append(candidate)
        if maraboupy_dir not in sys.path:
            sys.path.append(maraboupy_dir)

    try:
        from maraboupy import MarabouCore  # noqa: F401
        from maraboupy import MarabouNetworkNNet  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Failed to import the Marabou Python bindings required for ACASXu verification. "
            "Install maraboupy for this interpreter, or set NARV_MARABOU_DIR/MARABOU_DIR "
            "to a built Marabou repository. Tried: {}".format(
                ", ".join(tried_dirs) if tried_dirs else "<none>"
            )
        ) from exc

    if VERBOSE:
        print("Marabou import completed")
