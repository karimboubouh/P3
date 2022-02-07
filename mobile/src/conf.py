import socket

# LOCAL_PI = socket.gethostbyname(socket.gethostname())
LOCAL_PI = socket.gethostbyname('localhost')

TCP_SOCKET_BUFFER_SIZE = 500000
TCP_SOCKET_SERVER_LISTEN = 10
SOCK_TIMEOUT = 20
BRIDGE_HOST = "0.0.0.0"
BRIDGE_PORT = 15015
DEVICE_HOST = "0.0.0.0"
DEVICE_PORT = 0

ALGORITHM_MODULE = "p3"
WAIT_TIMEOUT = 10
WAIT_INTERVAL = 0.1
EVAL_ROUND = 10
VERBOSE = 0