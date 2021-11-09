import random

from numba import jit
from numba.core.registry import CPUDispatcher


@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


def monte_carlo_pi_2(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


disp = CPUDispatcher(monte_carlo_pi_2, locals={}, targetoptions={"nopython": True, "boundscheck": None})
args = tuple([disp.typeof_pyval(1)])
disp.compile(args)
print(monte_carlo_pi)

monte_carlo_pi(10)