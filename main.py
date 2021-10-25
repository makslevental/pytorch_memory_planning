import ctypes
import json
import os
from functools import partial

from example_models import make_bert
from plotting import make_memory_map
from profile import (
    get_required_mem_allocs,
    profile_model,
    get_mem_events,
)
from strategies import (
    greedy_by_size,
    find_gap,
    GapPriority,
    greedy_by_longest,
    bump_allocator,
    solve_cp,
)


def plan_greedy_strats(req_mem_allocs, name):
    planned_allocs = greedy_by_size(req_mem_allocs)
    make_memory_map(planned_allocs, f"greedy_by_size_with_smallest_gap_{name}")

    planned_allocs = greedy_by_size(
        req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    )
    make_memory_map(planned_allocs, f"greedy_by_size_with_first_gap_{name}")

    planned_allocs = greedy_by_longest(req_mem_allocs)
    make_memory_map(planned_allocs, f"greedy_by_longest_with_smallest_gap_{name}")

    planned_allocs = greedy_by_longest(
        req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    )
    make_memory_map(planned_allocs, f"greedy_by_longest_with_first_gap_{name}")


def make_maps(model, x, name):
    profile_model(model, x, name)
    trace_json = json.load(open(os.getcwd() + f"/traces/{name}.json"))
    req_mem_allocs = get_required_mem_allocs(trace_json)
    mem_events = get_mem_events(trace_json)

    make_memory_map(bump_allocator(mem_events), f"bump_{name}")
    plan_greedy_strats(req_mem_allocs, name)
    # make_memory_map(solve_mip(req_mem_allocs), f"mip_{name}")
    make_memory_map(solve_cp(req_mem_allocs), f"csp_{name}")


if __name__ == "__main__":
    for dir in ["models", "memory_maps", "traces"]:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    sh_obj = ctypes.cdll.LoadLibrary("runtime_patch/libruntime_patch.so")
    make_bert()
    # make_bert()
    # make_dcgan()
    # make_unet()
    #
    # name = "resnet18"
    # model = load_model(os.getcwd() + f"/models/{name}.pt")
    #
    # make_maps(model, torch.rand((1, 3, 100, 100)), "resnet18.1x3x100x100")
    # make_maps(model, torch.rand((2, 3, 100, 100)), "resnet18.2x3x100x100")
    # make_maps(model, torch.rand((4, 3, 100, 100)), "resnet18.4x3x100x100")
    # make_maps(model, torch.rand((8, 3, 100, 100)), "resnet18.8x3x100x100")
    # make_maps(model, torch.rand((16, 3, 100, 100)), "resnet18.16x3x100x100")
    # solve_z3()
    # hook_alloc_cpu()
