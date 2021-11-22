import sys

import pandas as pd
from matplotlib import pyplot as plt

# from typing import List
#
# import matplotlib.pyplot as plt
# import pandas as pd
# from torch import relu
from torch import relu

BACKEND = sys.argv[1]
NUM_ELEMENTS = int(sys.argv[2])


class MaxTensor:
    def __init__(self, ls=[]):
        self.ls = ls
        self.size = len(ls)

    def __add__(self, other):
        if BACKEND == "cpu":
            return add_cpu(self, other)
        elif BACKEND == "gpu":
            raise NotImplementedError

    def reset(self):
        del self.size, self.ls


def add_cpu(a: MaxTensor, b: MaxTensor) -> MaxTensor:
    assert a.size == b.size
    res = [0] * a.size
    for i in range(a.size):
        res[i] = a.ls[i] + b.ls[i]
    return MaxTensor(res)


# def relu(a: MaxTensor) -> MaxTensor:
#     res = [0] * a.size
#     for i, el in enumerate(a.ls):
#         res[i] = max(0, el)
#     return MaxTensor(res)


def add_relu(a: MaxTensor, b: MaxTensor) -> MaxTensor:
    assert a.size == b.size
    res = [0] * a.size
    for i in range(a.size):
        tmp = a.ls[i] + b.ls[i]
        res[i] = max(0, tmp)

    return MaxTensor(res)


def test_tensor(t1: MaxTensor, t2: MaxTensor):
    t1.size = t2.size = NUM_ELEMENTS
    t1.ls, t2.ls = list(range(NUM_ELEMENTS)), list(range(NUM_ELEMENTS))

    t3 = t1 + t2
    t4 = relu(t3)

    t1.reset(), t2.reset(), t3.reset(), t4.reset()


def plot_tensor_runs(fp):
    df = pd.read_csv(fp, delim_whitespace=True, header=None)
    dfs = []
    for i in range(1, 7):
        _df = df[i * 10 - 9 : i * 10]
        _df = _df.set_index(1)
        _df = _df.rename(columns={0: i})
        dfs.append(_df)

    concat_df = pd.concat(dfs, axis=1)
    concat_df = concat_df.apply(pd.to_numeric)
    concat_df.rename_axis(None, inplace=True)
    concat_df.drop(
        [
            "cycles",
            "context-switches",
            "cpu-migrations",
            "stalled-cycles-frontend",
            "stalled-cycles-backend",
            "branch-misses",
        ],
        inplace=True,
    )
    concat_T = concat_df.T
    concat_T.plot(figsize=(10, 6))

    ax = plt.gca()
    # ax.set_yscale('function', functions=(inverse, forward))
    plt.legend()

    ax.set_yscale("log")
    plt.xticks(range(1, 7), [f"$10^{i}$" for i in range(1, 7)])
    plt.xlabel("length")
    plt.title("perf stat for various MaxTensor lengths")
    plt.tight_layout()
    plt.savefig(f"{fp}.svg")
    plt.savefig(f"{fp}.pdf")
    print(concat_df)


def read_fusion(fp):
    df = pd.read_csv(fp, delim_whitespace=True, header=None)
    dfs = []
    for i in range(1, 10):
        _df = df[(i - 1) * 10 : i * 10]
        _df = _df.set_index(1)
        _df = _df.rename(columns={0: i})
        dfs.append(_df)

    concat_df = pd.concat(dfs, axis=1)
    concat_df = concat_df.apply(pd.to_numeric)
    concat_df.rename_axis(None, inplace=True)
    concat_df.drop(
        [
            "cycles",
            "context-switches",
            "cpu-migrations",
            "stalled-cycles-frontend",
            "stalled-cycles-backend",
            "branch-misses",
        ],
        inplace=True,
    )
    return concat_df


def plot_fused_runs(_fp):
    non_fused = read_fusion("non_fused.out")
    fused = read_fusion("fused.out")
    concat_df = non_fused / fused
    concat_T = concat_df.T
    concat_T.plot(figsize=(10, 6))

    ax = plt.gca()
    # ax.set_yscale('function', functions=(inverse, forward))
    plt.legend()

    # ax.set_yscale("log")
    plt.xticks(range(1, 10), [f"$10^{i}$" for i in range(1, 10)])
    plt.xlabel("length")
    plt.ylabel("%/100")
    plt.title("perf stat for non-fused/fused (various MaxTensor lengths)")
    plt.tight_layout()
    plt.savefig(f"nonfused_over_fused.svg")
    plt.savefig(f"nonfused_over_fused.pdf")


def plot_torchfused_runs(_fp):
    non_fused = read_fusion("mine_nonfused.out")
    fused = read_fusion("torch_nonfused.out")
    concat_df = non_fused / fused
    concat_T = concat_df.T
    concat_T.plot(figsize=(10, 6))

    ax = plt.gca()
    # ax.set_yscale('function', functions=(inverse, forward))
    plt.legend()

    # ax.set_yscale("log")
    plt.xticks(range(1, 9), [f"$10^{i}$" for i in range(1, 9)])
    plt.xlabel("length")
    plt.ylabel("%/100")
    plt.title("perf stat for PyTorch/MaxTensor non-fused (various MaxTensor lengths)")
    plt.tight_layout()
    plt.savefig(f"torch_nonfused_over_fused.svg")
    plt.savefig(f"torch_nonfused_over_fused.pdf")


def plot_slots():
    slots = list(map(float, open("slots.out").readlines()))
    no_slots = list(map(float, open("no_slots.out").readlines()))
    df = pd.DataFrame(
        {
            "slots": slots[:6],
            "no slots": no_slots[:6],
        }
    )
    df.plot()
    ax = plt.gca()
    ax.set_yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # t1 = torch.arange(0, NUM_ELEMENTS)
    # t2 = torch.arange(0, NUM_ELEMENTS)
    # t3 = t1 + t2
    # t4 = relu(t3)
    plot_torchfused_runs("")

    # t1 = MaxTensor()
    # t2 = MaxTensor()
    # test_tensor(t1, t2)
    #
    # for i in range(10):
    # fused_out = [f.strip().split() for f in open("fused.out").readlines()]
    # unfused_out = [f.strip().split() for f in open("non_fused.out").readlines()]
    # plot_fused_runs("fused.out")
