import os
import sys

from core.configuration.consts import CODE_DIR, VERBOSE


def _marabou_suffix(query_type):
    suffices_map = {
        "basic": "",
        "adversarial": "",
        "acas_xu_conjunction": "",
        # "basic": "_Reg",
        # "adversarial": "_Adv",
        # "acas_xu_conjunction": "_Adv"
    }
    if query_type not in suffices_map:
        raise KeyError("Unsupported Marabou query type '{}'.".format(query_type))
    return suffices_map[query_type]


def _candidate_marabou_dirs(query_type, explicit_marabou_dir=None):
    candidates = []
    for value in [
            explicit_marabou_dir,
            os.environ.get("NARV_MARABOU_DIR"),
            os.environ.get("MARABOU_DIR"),
            os.path.join(CODE_DIR, "Marabou{}".format(_marabou_suffix(query_type))),
    ]:
        if not value:
            continue
        candidate = os.path.abspath(os.fspath(value))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def dynamically_import_marabou(query_type="basic", marabou_dir=None):
    """
    dynamically import the relevant marabou version a.t. property type
    :param property_type: Str, type of the query
    """
    tried_dirs = _candidate_marabou_dirs(query_type, explicit_marabou_dir=marabou_dir)
    for marabou_dir_candidate in tried_dirs:
        maraboupy_dir = os.path.join(marabou_dir_candidate, "maraboupy")
        if marabou_dir_candidate not in sys.path:
            sys.path.append(marabou_dir_candidate)
        if maraboupy_dir not in sys.path:
            sys.path.append(maraboupy_dir)

    # verity that the import works
    try:
        from maraboupy import MarabouCore
        from maraboupy import MarabouNetworkNNet as mnn
    except ModuleNotFoundError as exc:
        if exc.name != "maraboupy":
            raise
        raise ModuleNotFoundError(
            "Failed to import 'maraboupy'. Install Marabou for the Python interpreter "
            "running this script, or set NARV_MARABOU_DIR/MARABOU_DIR to the Marabou "
            "repository directory. Tried Marabou directories: {}".format(
                ", ".join(tried_dirs) if tried_dirs else "<none>"
            )
        ) from exc
    if VERBOSE:
        print(f"sys.path={sys.path}")
        print("finish import marabou")

# if __name__ == "__main__":
#     dynamically_import_marabou()
