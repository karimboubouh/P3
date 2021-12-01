import pickle

CONNECT = 0
DISCONNECT = 1
TRAIN_STEP = 3
TRAIN_STOP = 4


def connect(address, node_id):
    return pickle.dumps({
        'mtype': CONNECT,
        'data': {'address': address, 'id': node_id},
    })


def disconnect(node_id):
    return pickle.dumps({
        'mtype': DISCONNECT,
        'data': {'id': node_id},
    })


def train_step(t, update):
    return pickle.dumps({
        'mtype': TRAIN_STEP,
        'data': {'t': t, 'update': update},
    })


def stop_train():
    return pickle.dumps({
        'mtype': TRAIN_STOP,
        'data': {},
    })
