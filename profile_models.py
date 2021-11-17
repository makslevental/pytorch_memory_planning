from collections import defaultdict

import numpy as np
import torch
from torch._C._autograd import ProfilerActivity
from torch.profiler import profile
# from torch.utils import cpp_extension

from strategies import LiveRange, MemEvent, RequiredAlloc


def register_op():
    op_source = open("profile_models.cpp").read()
    cpp_extension.load_inline(
        name="prep_test_models",
        cpp_sources=op_source,
        # extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
        is_python_module=False,
        verbose=True,
    )
    print(torch.ops.my_ops.prep_test_models)


# register_op()


def profile_model(model, x, trace_f_name):
    with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
    ) as prof:
        with torch.no_grad():
            model(x)
    print(model.inlined_graph)
    prof.export_chrome_trace(f"traces/{trace_f_name}.json")


def analyze_model(model, x):
    if not isinstance(x, tuple):
        x = (x,)
    model_frozen = torch.jit.freeze(torch.jit.script(model.eval()))
    model_frozen.graph.eraseInput(0)
    live_ranges = []
    with torch.jit._hide_source_ranges():
        g = torch._C._jit_trace_graph(model_frozen.graph, x)
        node_to_idx = {node: i for i, node in enumerate(g.nodes())}
        for node in g.nodes():
            # if node.kind() == "prim::Constant": continue
            if node.kind() == "prim::NumToTensor": continue
            if node.kind() == "aten::slice": continue
            if node.kind() == "aten::expand": continue
            if node.kind() == "aten::unsqueeze": continue
            users = [u.user.kind() for u in node.output().uses()]
            if "prim::Return" in users: continue

            if node.output().type().kind() == 'TensorType':
                sizes = node.output().type().symbolic_sizes()
                type = node.output().type().str().split("(")[0]
                if type != "Float":
                    # print(node.output())
                    continue
                end = start = node_to_idx[node]
                for u in node.output().uses():
                    idx = node_to_idx[u.user]
                    end = max(end, idx)

                live_ranges.append(
                    RequiredAlloc(LiveRange(start, end), int(np.prod(sizes))*4, node.output().debugName())
                )
    mem_events = []
    for lvr in live_ranges:
        mem_events.append(
            MemEvent(
                lvr.ptr_addr,
                lvr.size,
                lvr.lvr.begin
            )
        )
        mem_events.append(
            MemEvent(
                lvr.ptr_addr,
                -lvr.size,
                lvr.lvr.end+1
            )
        )
    return live_ranges, mem_events

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
        "models/resnet18.pt"
    )
    # profile_model(model, torch.rand((1, 3, 100, 100)), "resnet18.1x3x100x100")
    # profile_model(model, torch.rand((1, 3, 200, 100)), "resnet18.1x3x200x100")
    profile_model(model, torch.rand((1, 3, 100, 200)), "resnet18.1x3x100x200")
    profile_model(model, torch.rand((1, 3, 100, 200)), "resnet18.1x3x100x200")
    # trace_json = json.load(
    #     open("/Users/mlevental/dev_projects/pytorch_shape_inference/shape_inference/traces/resnet18.json"))
    # mem_events = get_mem_events(trace_json)
    # required_mem_allocs = get_required_mem_allocs(trace_json)
    # print(mem_events)
