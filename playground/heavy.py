import os

from src.measure_energy import measure_energy
from src.utils import log


@measure_energy
def heavy(load=25, timeout=10):
    os.system(f"stress -c {load}  -t {timeout}")


def correct(a, b, c, d, e, f, g, h):
    zz = (a + b + c + d) / 4
    x = e - zz
    print(round(e - x, 2))
    print(round(f - x, 2))
    print(round(g - x, 2))
    print(round(h - x, 2))


if __name__ == '__main__':
    heavy(timeout=25)
