import sys
from termcolor import cprint
import numpy as np

from src.helpers import Map


def log(mtype, message, title=True):
    global args
    if args.verbose > -2:
        if mtype == "result":
            cprint(" Result:  ", 'blue', attrs=['reverse'], end=' ')
            cprint(message, 'blue')
            log.color = 'blue'
            return
    if args.verbose > -1:
        if mtype == "error":
            cprint(" Error:   ", 'red', attrs=['reverse'], end=' ')
            cprint(message, 'red')
            log.color = 'red'
            return
        elif mtype == "success":
            cprint(" Success: ", 'green', attrs=['reverse'], end=' ')
            cprint(message, 'green')
            log.color = 'green'
            return
    if args.verbose > 0:
        if mtype == "event":
            cprint(" Event:   ", 'magenta', attrs=['reverse'], end=' ')
            cprint(message, 'magenta')
            log.color = 'magenta'
            return
        elif mtype == "warning":
            cprint(" Warning: ", 'yellow', attrs=['reverse'], end=' ')
            cprint(message, 'yellow')
            log.color = 'yellow'
            return
    if args.verbose > 1:
        if mtype == "info":
            cprint(" Info:    ", 'grey', attrs=['reverse'], end=' ')
            cprint(message, 'grey')
            log.color = 'grey'
            return
    cprint(" Log:     ", 'cyan', attrs=['reverse'], end=' ')
    cprint(message, 'cyan')
    log.color = 'cyan'


if __name__ == '__main__':
    args = Map()
    args.verbose = 10
    m = "Operation completed"
    log("result", m)
    log("error", m)
    log("success", m)
    log("event", m)
    log("warning", m)
    log(None, "my message")
    log("info", m)
    log("log", m)
    log("log", m)
    log("log", m)
    log("event", m)
