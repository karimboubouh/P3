import socket

LOCAL_PI = socket.gethostbyname(socket.gethostname())
# LOCAL_PI = socket.gethostbyname('localhost')
# LOCAL_PI = ''
HOST = LOCAL_PI
PORT = 9000
LAUNCHER_HOST = LOCAL_PI
LAUNCHER_PORT = 15015
TCP_SOCKET_BUFFER_SIZE = 500000
TCP_SOCKET_SERVER_LISTEN = 10
SOCK_TIMEOUT = 20
LAUNCHER_TIMEOUT = 60

ML_ENGINE = "NumPy"  # "NumPy" or "PyTorch"
DEFAULT_VAL_DS = "val"
DEFAULT_MEASURE = "mean"
EVAL_ROUND = 10
TRAIN_VAL_TEST_RATIO = [.8, .1, .1]
RECORD_RATE = 10
M_CONSTANT = 1
WAIT_TIMEOUT = 1.5
WAIT_INTERVAL = 0.02
FUNC_TIMEOUT = 60
# TODO check TEST_SCOPE
TEST_SCOPE = 'neighborhood'
IDLE_POWER = 12.70
INFERENCE_BATCH_SIZE = 512
DATASET_DUPLICATE = 0
