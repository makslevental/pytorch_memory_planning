import os
from collections import defaultdict
from operator import itemgetter
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches

from strategies import PlannedAlloc


def get_cmap(n, name="spring"):
    return plt.cm.get_cmap(name, n)


def add_envelope_to_memory_map(
        fig,
        ax,
        x_min,
        _x_max,
        _y_min,
        _y_max,
        allocations: List[PlannedAlloc],
        title,
        *,
        shade=True,
        save=True,
        fp_dir=os.getcwd() + "/memory_maps",
        rescale_to_mb=True,
):
    envelope_x = []
    envelope_y = []
    allocations.sort(key=lambda a: a.lvr.end)
    for i, alloc in enumerate(allocations):
        begin, end, offset, size = (
            alloc.lvr.begin,
            alloc.lvr.end,
            alloc.mem_region.offset,
            alloc.mem_region.size,
        )

        if offset + size > max(
                [allo.mem_region.offset + allo.mem_region.size for allo in allocations[i:]],
                default=float("inf"),
        ):
            envelope_x.append(end)
            if rescale_to_mb:
                envelope_y.append((offset + size) / 2 ** 20)
            else:
                envelope_y.append(offset + size)

    if envelope_y:
        envelope_x.insert(0, x_min)
        envelope_y.insert(0, envelope_y[0])
    else:
        print("no envelope")
        return

    line = ax.plot(envelope_x, envelope_y, marker="o", ls="--", color="r")

    if save:
        assert fp_dir is not None
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_envelope.pdf")
        # fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_envelope.svg")

    if not shade:
        return fig, ax, x_min, _x_max, _y_min, _y_max

    ax.lines.remove(line[0])
    for i, x in enumerate(envelope_x[1:], start=1):
        ax.vlines(x=x, ymin=0, ymax=envelope_y[i], colors="r", ls="--")

    ax.set_xticks(envelope_x[1:])
    ax.set_xticklabels(
        [f"{envelope_y[i + 1]:.3f} mb" for i in range(len(envelope_x[1:]))], rotation=90
    )
    ax.fill_between(envelope_x, envelope_y, step="pre", alpha=0.4, zorder=100)

    fig.tight_layout()

    if save:
        assert fp_dir is not None
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_shading.pdf")
        # fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_shading.svg")

    return fig, ax, x_min, _x_max, _y_min, _y_max


def _make_memory_map(
        allocations: List[PlannedAlloc],
        title,
        *,
        save=True,
        fp_dir=os.getcwd() + "/memory_maps",
        rescale_to_mb=True,
):
    fig, ax = plt.subplots(figsize=(50, 10))

    x_min, x_max, y_min, y_max = float("inf"), 0, float("inf"), 0
    cmap = get_cmap(len(allocations))
    for i, alloc in enumerate(allocations):
        begin, end, offset, size = (
            alloc.lvr.begin,
            alloc.lvr.end,
            alloc.mem_region.offset,
            alloc.mem_region.size,
        )
        if rescale_to_mb:
            x, y = begin, (offset / 2 ** 20)
            width, height = end - begin, (size / 2 ** 20)
        else:
            x, y = begin, offset
            width, height = end - begin, size

        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x_max, x + width)
        y_max = max(y_max, y + height)

        if "mip" in title or "csp" in title:
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=0.5,
                facecolor="green",
                edgecolor="black",
            )
        else:
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=0.5,
                edgecolor="black",
                facecolor=cmap(i),
            )
        ax.add_patch(rect)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("time")
    ax.set_ylabel(f"memory ({'mb' if rescale_to_mb else 'b'})")
    ax.set_title(title)

    fig.tight_layout()
    if save:
        assert fp_dir is not None
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}.pdf")
        # fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}.svg")

    return fig, ax, x_min, x_max, y_min, y_max


def make_memory_map(
        allocations: List[PlannedAlloc],
        title,
        *,
        save=True,
        fp_dir=os.getcwd() + "/memory_maps",
        rescale_to_mb=True,
        add_envelope=False,
):
    fig, ax, x_min, x_max, y_min, y_max = _make_memory_map(
        allocations, title, save=save, fp_dir=fp_dir, rescale_to_mb=rescale_to_mb
    )
    if add_envelope:
        add_envelope_to_memory_map(
            fig,
            ax,
            x_min,
            x_max,
            y_min,
            y_max,
            allocations,
            title,
            shade=True,
            save=save,
            fp_dir=fp_dir,
            rescale_to_mb=rescale_to_mb,
        )
    plt.close(fig)


strat_colors = {
    "greedy_by_breadth": "red",
    "greedy_by_longest_and_size_with_first_gap": "blue",
    "greedy_by_longest_and_size_with_smallest_gap": "green",
    "greedy_by_size_with_first_gap": "orange",
    "greedy_by_size_with_smallest_gap": "pink",
    "mip": "purple",
    "naive": "black",
    "linear_scan": "gray",
    "eager": "teal",
    "profiled greedy_by_breadth": "red",
    "profiled greedy_by_longest_and_size_with_first_gap": "blue",
    "profiled greedy_by_longest_and_size_with_smallest_gap": "green",
    "profiled greedy_by_size_with_first_gap": "orange",
    "profiled greedy_by_size_with_smallest_gap": "pink",
    "profiled mip": "purple",
    "profiled naive": "black",
}


def plot_mem_usage(
        mem_usage,
        title,
        normalizer_name="mip",
        logy=False,
        last_one="linear_scan",
        ylabel="% max mem",
):
    strategies = set()
    for model, strats in mem_usage.items():
        if "naive" in strats:
            strats.pop("naive")
        if "profiled naive" in strats:
            strats.pop("profiled naive")
        strategies = strategies.union(strats.keys())
    labels = list(mem_usage.keys())
    xs = (len(strategies) + 5) * np.arange(len(labels))  # the label locations
    label_to_x = dict(zip(labels, xs))
    width = 1  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rectss = defaultdict(dict)
    for model, strat_results in mem_usage.items():
        normalizer = strat_results[normalizer_name]
        if last_one in strat_results:
            last_one_res = strat_results.pop(last_one)
            strat_results[last_one] = last_one_res
        for i, (strat, mem) in enumerate(strat_results.items()):
            mem /= normalizer
            x = label_to_x[model] - len(strategies) // 2 + i
            rectss[strat][model] = (x, mem)

    rects = {}
    if last_one in rectss:
        last_one_res = rectss.pop(last_one)
        rectss[last_one] = last_one_res
    for strat, models in rectss.items():
        print(strat, [f"{m[1]:.2f}" for m in models.values()])
        rects[strat] = ax.bar(
            [m[0] for m in models.values()],
            [m[1] for m in models.values()],
            width,
            color=strat_colors[strat],
            label=strat.replace("profiled ", "")
                .replace("static ", "")
                .replace("_", " ")
                .upper(),
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if logy:
        ax.set_yscale("log")
        # ax.set_ylim(1)
    # ax.set_ylabel(("log " if logy else "") + ylabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75)
    ax.legend(bbox_to_anchor=(0.55, 1.1), loc="lower left")

    for strat in strategies:
        models = rectss[strat]
        labels = [f"{m[1]:.2f}" for m in models.values()]
        ax.bar_label(rects[strat], labels, padding=3, rotation=90)

    fig.tight_layout()

    fig.savefig(
        f"/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/{title.replace(' ', '_')}.pdf"
    )
    fig.savefig(
        f"/home/mlevental/dev_projects/pytorch_dev/memory_allocator/benchmarks/{title.replace(' ', '_')}.svg"
    )


# sns.set(rc={'figure.figsize':(20, 10)})


def plot_results(fp):
    df = pd.read_csv(fp)
    piv = df.pivot(index="model", columns="strategy", values="size")
    piv /= 2 ** 20
    piv.drop(["Bump Allocator"], axis=1, inplace=True)
    # ax = sns.barplot(x="model", y="size", hue="strategy", data=df)
    # ax = sns.barplot(x=piv.index, y="size", hue="strategy", data=piv)

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    # for a in [ax, ax2]:
    #     a.majorticks_on()
    #     a.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    #     a.minorticks_on()
    #     a.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    piv.loc[
        [
            "inception_v3",
            "resnet152",
            "resnext50_32x4d",
            "vgg13",
            "deeplabv3_resnet50",
            "densenet121",
            "densenet161",
        ]
    ].plot.bar(rot=60, figsize=(12, 7), ax=ax)
    piv.loc[
        [
            "inception_v3",
            "resnet152",
            "resnext50_32x4d",
            "vgg13",
            "deeplabv3_resnet50",
            "densenet121",
            "densenet161",
        ]
    ].plot.bar(rot=60, figsize=(12, 7), ax=ax2)
    ax.set_ylim(530, 553)  # outliers only
    ax2.set_ylim(0, 70)  # most of the data
    f.subplots_adjust(hspace=0.05)  # adjust space between axes

    # hide the spines between ax and ax2
    ax.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=22,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax.plot([0.5], [0], transform=ax.transAxes, **kwargs)
    ax2.plot([0.5], [1], transform=ax2.transAxes, **kwargs)

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'
    ax2.get_legend().remove()
    ax2.set_ylabel("Size (MB)", fontsize=24)
    ax2.yaxis.set_label_coords(-0.05, 1.02)

    num_bars_in_group = len(ax.containers[0].patches)
    for i in range(num_bars_in_group):
        bar_group = list(map(itemgetter(i), ax.containers))
        xmin = min(bar_group, key=lambda b: b.get_x()).get_x()
        xmax = max(bar_group, key=lambda b: b.get_x() + b.get_width())
        xmax = xmax.get_x() + xmax.get_width()
        y = min(bar_group, key=lambda b: b.get_height()).get_height()
        print(xmin, xmax, y)
        ax.hlines(
            y=y, xmin=xmin, xmax=xmax, linewidth=0.5, linestyle="--", color="black"
        )
        if i == 3:
            pass
            # ax.plot([(xmax+xmin)/2], [0], transform=ax.transAxes, **kwargs)
            # ax2.plot([(xmax+xmin)/2], [1], transform=ax2.transAxes, **kwargs)

    for i in range(num_bars_in_group):
        bar_group = list(map(itemgetter(i), ax2.containers))
        xmin = min(bar_group, key=lambda b: b.get_x()).get_x()
        xmax = max(bar_group, key=lambda b: b.get_x() + b.get_width())
        xmax = xmax.get_x() + xmax.get_width()
        y = min(bar_group, key=lambda b: b.get_height()).get_height()
        print(xmin, xmax, y)
        ax2.hlines(
            y=y, xmin=xmin, xmax=xmax, linewidth=0.5, linestyle="--", color="black"
        )

    ax2.set_xlabel("Model", labelpad=-40, fontsize=24)
    ax2.xaxis.set_tick_params(labelsize=14)

    leg = ax.legend(fontsize=14)
    leg.set_title("Strategy", prop={"size": 24})
    # ax.set_yscale("log")
    # plt.tight_layout()
    #
    # # Show the major grid lines with dark grey lines
    #
    # # plt.show()
    f.tight_layout()
    f.savefig("memory_bench.pdf")
    # f.show()


def plot_jemalloc_heap_profile(fp, model_name):
    df = pd.read_csv(fp, names="type,ts,ptr,allocated,active,metadata,resident,mapped,retained".split(","))
    df["allocated,active,metadata,resident,mapped,retained".split(",")] /= 2 ** 20
    ax = df["allocated,active,metadata,resident,mapped,retained".split(",")].plot(figsize=(12, 7))
    ax.set_xlabel("Model", labelpad=10, fontsize=24)
    ax.set_ylabel("MB", labelpad=10, fontsize=24)
    ax.set_title(f"{model_name} jemalloc stats", fontsize=24)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"/home/mlevental/dev_projects/pytorch_memory_planning/jemalloc_plots/{model_name}.pdf")
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("/home/mlevental/dev_projects/pytorch_memory_planning/je_malloc_runs/res.csv", names=["model", "normal", "uptime_ns"])
    piv = df.pivot(index="model", columns="normal", values="uptime_ns")
    piv = piv.dropna()
    print(df)
    runs_dir = "/home/mlevental/dev_projects/pytorch_memory_planning/je_malloc_runs/"
    for path, directories, files in os.walk(runs_dir):
        _, name = os.path.split(path)
        if "heap_profile.csv" in set(files):
            plot_jemalloc_heap_profile(f"{path}/heap_profile.csv", name)
