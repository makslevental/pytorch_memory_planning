import os
from typing import List

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
            alloc.offset,
            alloc.size,
        )

        if offset + size > max(
            [allo.offset + allo.size for allo in allocations[i:]],
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
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_envelope.svg")

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
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}_with_shading.svg")

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
            alloc.offset,
            alloc.size,
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
        fig.savefig(f"{fp_dir}/{title.replace(' ', '_')}.svg")

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
