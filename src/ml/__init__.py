from src.conf import ML_ENGINE

if ML_ENGINE.lower() == "pytorch":
    from src.ml.pytorch.models import *
    from src.ml.pytorch.helpers import *
    from src.ml.pytorch.datasets import get_dataset, train_val_test, inference_ds
elif ML_ENGINE.lower() == "numpy":
    from src.ml.numpy.models import *
    from src.ml.numpy.helpers import *
    from src.ml.numpy.datasets import get_dataset, train_val_test, inference_ds
else:
    exit(f'Unknown "{ML_ENGINE}" ML engine !')
