import json
from collections import defaultdict

from torch._C._autograd import ProfilerActivity

import torch
from strategies import LiveRange, MemEvent, RequiredAlloc
from torch.profiler import profile


def make_resnets():
    models_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    for models_name in models_names:
        model = torch.hub.load("pytorch/vision:v0.10.0", models_name)
        model.eval()
        x = torch.rand((1, 3, 224, 244))
        y = torch.jit.trace(model, (x,))

        y.save(f"models/{models_name}.pt")


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
    # model = load_model(
    #     "/Users/mlevental/dev_projects/pytorch_shape_inference/shape_inference/models/resnet34.pt"
    # )
    # profile_model(model, torch.rand((1, 3, 224, 244)), "resnet34")
    trace_json = json.load(
        open("/Users/mlevental/dev_projects/pytorch_shape_inference/shape_inference/traces/resnet18.json"))
    mem_events = get_mem_events(trace_json)
    required_mem_allocs = get_required_mem_allocs(trace_json)
    print(mem_events)
