import importlib

from src.conf import ML_ENGINE
from src.utils import log


def initialize_models(args, same=False):
    if ML_ENGINE.lower() == "pytorch":
        ml = importlib.import_module("src.model.models_pytorch")
    elif ML_ENGINE.lower() == "numpy":
        ml = importlib.import_module("src.model.models_numpy")
    else:
        log('error', f"Unknown ML engine <{ML_ENGINE}>")
        exit(0)

    return ml.initialize_models(args, same=same)
