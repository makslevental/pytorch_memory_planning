from collections import defaultdict

import torch
from torch._C._autograd import ProfilerActivity
from torch.profiler import profile

from strategies import LiveRange, MemEvent, RequiredAlloc


def profile_model(model, x, trace_f_name):
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            model(x)
    prof.export_chrome_trace(f"traces/{trace_f_name}.json")


def load_model(fp):
    model = torch.jit.load(fp)
    return model


def get_mem_events(trace_json):
    mem_events = []
    seen_ptr_addrs = defaultdict(lambda: 0)

    for trace_evt in trace_json["traceEvents"]:
        if trace_evt["name"] == "[memory]":
            ptr_addr = trace_evt["args"]["Addr"]
            # same ptr addrs might be reused
            # make ptr pairs unique by incrementing each pair
            unique_ptr_addr = f"{ptr_addr}_{seen_ptr_addrs[ptr_addr] // 2}"
            seen_ptr_addrs[ptr_addr] += 1

            size = trace_evt["args"]["Bytes"]
            ts = trace_evt["ts"]
            mem_events.append(MemEvent(unique_ptr_addr, size, ts))

    return mem_events


def get_required_mem_allocs(trace_json):
    mem_events_paired = defaultdict(list)
    for (ptr_addr, size, ts) in get_mem_events(trace_json):
        mem_events_paired[ptr_addr].append((size, ts))

    live_ranges = []
    for ptr_addr, (alloc_evt, free_evt) in mem_events_paired.items():
        alloc_size, alloc_ts = alloc_evt
        free_size, free_ts = free_evt
        assert alloc_size == -free_size
        assert alloc_ts < free_ts
        live_ranges.append(
            RequiredAlloc(LiveRange(alloc_ts, free_ts), alloc_size, ptr_addr)
        )
    return live_ranges


if __name__ == "__main__":
    # make_resnets()
    model = load_model(
        "/Users/mlevental/dev_projects/pytorch_memory_planning/models/resnet18.pt"
    )
    # profile_model(model, torch.rand((1, 3, 100, 100)), "resnet18.1x3x100x100")
    # profile_model(model, torch.rand((1, 3, 200, 100)), "resnet18.1x3x200x100")
    profile_model(model, torch.rand((1, 3, 100, 200)), "resnet18.1x3x100x200")
    profile_model(model, torch.rand((2, 3, 100, 100)), "resnet18.2x3x100x100")
    # trace_json = json.load(
    #     open("/Users/mlevental/dev_projects/pytorch_shape_inference/shape_inference/traces/resnet18.json"))
    # mem_events = get_mem_events(trace_json)
    # required_mem_allocs = get_required_mem_allocs(trace_json)
    # print(mem_events)
