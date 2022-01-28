"""
file    : models.py
desc    : contains models implementations
classes : - MLP
          - CNNMnist
          - CNNFashion_Mnist
          - CNNCifar
          - ModelBased
"""
import copy

from src.model.Model import *
from src.utils import log


def initialize_models(args, same=False):
    models = []
    architecture = None
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            architecture = CNNMnist
        elif args.dataset == 'fmnist':
            pass
        elif args.dataset == 'cifar':
            pass
    elif args.model == 'mlp':
        # Multi-layer perceptron
        if args.dataset == 'mnist':
            architecture = FFNMnist
        elif args.dataset == 'cifar':
            log('error', f"Model <MLP> is not compatible with <CIFAR> dataset.")
            exit(0)
        else:
            architecture = MLP
    elif args.model == 'linear':
        architecture = LogisticRegression
    else:
        exit('Error: unrecognized model')

    # model params
    name = args.model.upper()
    if args.model in ['cnn', 'mlp']:
        input_dim = [28, 28, 1]
    else:
        input_dim = [28, 28]

    if same:
        # Initialize all models with same weights
        model = Model(name=name, input_dim=input_dim).initial(architecture)
        for i in range(args.num_users):
            models.append(copy.deepcopy(model))
        return models

    else:
        # Independent initialization
        for i in range(args.num_users):
            model = Model(name=name, input_dim=input_dim).initial(architecture)
            models.append(model)

    print(models)
    exit()
    return models


# ====> args.model: mlp | args.dataset: mnist ---------------------------------
FFNMnist = [
    Flatten(name='flatten'),
    Dense(name='fc1', units=100),
    Activation(name='A1', method='relu'),
    Dense(name='fc2', units=100),
    Activation(name='A2', method='relu'),
    Dense(name='fc3', units=10),
    Activation(name='A3', method='softmax'),
]

# ====> args.model: cnn | args.dataset: mnist ---------------------------------

CNNMnist = [
    Conv2D(name='C1', kernel_size=[3, 3], filters=5, padding='valid'),
    Activation(name='A1', method='relu'),
    MaxPooling2D(name='P1', pooling_size=[2, 2]),
    Flatten(name='flatten'),
    Dense(name='fc1', units=100),
    Activation(name='A3', method='relu'),
    Dense(name='fc2', units=10),
    Activation(name='A4', method='softmax'),
]

# ====> args.model: rnn | args.dataset: mnist ---------------------------------

RNNMnist = [
    BasicRNN(name='R1', units=10, return_last_step=False),
    Activation(name='A1', method='relu'),
    BasicRNN(name='R2', units=10, return_last_step=False),
    Activation(name='A2', method='relu'),
    Flatten(name='flatten'),
    Dense(name='fc1', units=10),
    Activation(name='A3', method='softmax'),
]
