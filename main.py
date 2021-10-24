import json
import os
from functools import partial

import torch

from example_models import make_unet, make_bert, make_dcgan
from plotting import make_memory_map
from profile import (
    get_required_mem_allocs,
    profile_model,
    get_mem_events,
    load_model,
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


# load funchook
import ctypes


def hook_alloc_cpu():
    # should work on python 2.7/3 windows/linux

    fh_lib = ctypes.cdll.LoadLibrary(
        "/Users/mlevental/dev_projects/funchook/build/libfunchook.dylib"
    )
    pytorch_malloc_lib = ctypes.cdll.LoadLibrary(
        "/Users/mlevental/dev_projects/pytorch_shape_inference/cmake-build-debug/lib/libpytorch_memory_allocator.dylib"
    )

    # define signatures
    funchook_create = fh_lib.funchook_create
    funchook_create.restype = ctypes.c_void_p
    funchook_create.argtypes = []

    funchook_prepare = fh_lib.funchook_prepare
    funchook_prepare.restype = ctypes.c_ssize_t
    funchook_prepare.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    funchook_install = fh_lib.funchook_install
    funchook_install.restype = ctypes.c_ssize_t
    funchook_install.argtypes = [ctypes.c_void_p, ctypes.c_int]

    alloc_cpu = pytorch_malloc_lib.alloc_cpu
    alloc_cpu.restype = ctypes.c_void_p
    alloc_cpu.argtypes = [ctypes.c_ssize_t]

    free_cpu = pytorch_malloc_lib.free_cpu
    free_cpu.restype = None
    free_cpu.argtypes = [ctypes.c_void_p]

    # must keep those references alive, or stuff will be GC'd and weird errors will occur
    global orig_write, hook, orig_write_ptr

    # create hook (this function will replace the original function)
    hook_type = ctypes.PYFUNCTYPE(None, ctypes.c_char_p)
    orig_write = None

    def hook_impl(msg):
        print("about to write: " + str(msg))  # do what we want
        orig_write(msg)  # call the original function

    hook = hook_type(hook_impl)

    fh = funchook_create()
    # create a pointer object with the function address
    orig_write_ptr = ctypes.c_void_p(
        ctypes.c_void_p.from_address(ctypes.addressof(alloc_cpu)).value
    )
    # orig_write_ptr.value will get a ptr to the original PySys_WriteStdout and PySys_WriteStdout will now point to the hook
    ret = funchook_prepare(fh, ctypes.addressof(orig_write_ptr), hook)
    assert not ret, "ret is " + str(ret)
    ret = funchook_install(fh, 0)
    assert not ret, "ret is " + str(ret)
    orig_write = hook_type.from_address(ctypes.addressof(orig_write_ptr))
    alloc_cpu(b"hi there\n")


if __name__ == "__main__":
    for dir in ["models", "memory_maps", "traces"]:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    make_bert()
    make_dcgan()
    make_unet()

    name = "resnet18"
    model = load_model(os.getcwd() + f"/models/{name}.pt")

    make_maps(model, torch.rand((1, 3, 100, 100)), "resnet18.1x3x100x100")
    make_maps(model, torch.rand((2, 3, 100, 100)), "resnet18.2x3x100x100")
    make_maps(model, torch.rand((4, 3, 100, 100)), "resnet18.4x3x100x100")
    make_maps(model, torch.rand((8, 3, 100, 100)), "resnet18.8x3x100x100")
    make_maps(model, torch.rand((16, 3, 100, 100)), "resnet18.16x3x100x100")
    # solve_z3()
    # hook_alloc_cpu()
