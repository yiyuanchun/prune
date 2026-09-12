import os
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
CODE_DIR = os.environ.get("NARV_CODE_DIR", WORKSPACE_ROOT)

PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "ACASXu",
    "networks",
)
PATH_TO_MARABOU_ACAS_EXAMPLES = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
results_directory = os.path.join(PROJECT_ROOT, "results")

VERBOSE = False
INT_MAX = sys.maxsize
INT_MIN = -sys.maxsize - 1
SAT_EXIT_CODE = 1
UNSAT_EXIT_CODE = 2

INPUT_LOWER_BOUND = -0.5
INPUT_UPPER_BOUND = 0.5

DO_ABSTRACTION_TO_FIRST_HIDDEN_LAYER = False
if DO_ABSTRACTION_TO_FIRST_HIDDEN_LAYER:
    FIRST_ABSTRACT_LAYER = 1
    FIRST_INC_DEC_LAYER = 0
    FIRST_POS_NEG_LAYER = 0
else:
    FIRST_ABSTRACT_LAYER = 2
    FIRST_INC_DEC_LAYER = 1
    FIRST_POS_NEG_LAYER = 1

COMPLETE_ABSTRACTION = True
VISUAL_WEIGHT_CONST = 3
LAYER_INTERVAL = 8
NODE_INTERVAL = 2
LAYER_TYPE_NAME2COLOR = {
    "input": "r",
    "hidden": "b",
    "output": "g",
}
SORTING_COLOR_MAP = {
    "white": 0,
    "r": 1,
    "b": 2,
    "g": 3,
}

EPSILON = 10 ** -5
DELTA = 0.01
PROPERTY_ID = "property_3"
COMPARE_TO_PREPROCESSED_NET = False
cur_time_str = "_".join(str(datetime.now()).rpartition(".")[0].split(" "))
ar_type2sign = {
    "inc": "+",
    "dec": "-",
}
