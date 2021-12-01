import os

from .measure_energy import measure_energy
from src.utils import log


@measure_energy
def heavy(load=25, timeout=10):
    os.system(f"stress -c {load}  -t {timeout}")


if __name__ == '__main__':
    heavy(timeout=25)
