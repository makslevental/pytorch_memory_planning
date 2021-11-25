import argparse
import ctypes
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from pprint import pprint
from typing import List

import numpy as np

import torch
from torch._C._autograd import ProfilerActivity
from torch.profiler import profile

from example_models import vision_models, make_resnets
from strategies import LiveRange, MemEvent, RequiredAlloc


def register_op():
    from torch.utils import cpp_extension
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

def get_reqs_from_trace(name):
    trace_json = json.load(open(os.getcwd() + f"/traces/{name}.json"))
    req_mem_allocs = get_required_mem_allocs(trace_json)
    mem_events = get_mem_events(trace_json)
    return req_mem_allocs, mem_events


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
    # _disable_profiler()


def analyze_model(model, x):
    torch._C.Graph.set_global_print_source_ranges(False)
    if not isinstance(x, tuple):
        x = (x,)
    model_frozen = torch.jit.freeze(torch.jit.trace(model.eval(), x, strict=False))
    # model_frozen = torch.jit.freeze(torch.jit.script(model.eval()))
    graph = model_frozen.graph
    self = list(graph.inputs())[0]
    if "self" in self.debugName() and not self.uses():
        graph.eraseInput(0)

    graph = torch._C._jit_trace_graph(graph, x)
    torch._C._jit_pass_optimize_frozen_graph(graph)
    for node in graph.findAllNodes("prim::RaiseException"):
        node.destroy()
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_remove_mutation(graph)
    torch._C._jit_pass_peephole(graph)

    req_allocs = []
    def make_alloc(node):
        for outp in node.outputs():
            users = [u.user.kind() for u in outp.uses()]
            if "prim::Return" in users:
                return
        # print(node, node.outputsSize(), node.output().type().kind())
        if node.outputsSize() != 1 or node.output().type().kind() != "TensorType":
            return
        if node.kind() == "prim::Constant":
            return
        if node.kind() == "prim::VarConcat":
            return
        if node.kind() == "prim::ListConstruct":
            return
        if node.kind() == "prim::TupleConstruct":
            return
        # if node.kind() == "prim::ListUnpack":
        #     return
        # if node.kind() == "prim::NumToTensor":
        #     return
        # if node.kind() == "aten::gt":
        #     return
        # if node.kind() == "aten::eq":
        #     return
        # if node.kind() == "aten::len":
        #     return
        # if node.kind() == "aten::size":
        #     return
        # if node.kind() == "aten::warn":
        #     return
        # if node.kind() == "aten::dim":
        #     return
        # if node.kind() == "aten::slice":
        #     return
        # if node.kind() == "aten::expand":
        #     return
        # if node.kind() == "aten::unsqueeze":
        #     return

        if node.outputsSize() == 1 and node.output().type().kind() == "TensorType":
            sizes = node.output().type().symbolic_sizes()
            type = node.output().type().str().split("(")[0]
            if type != "Float":
                return
            end = start = node_to_idx[node]
            for u in node.output().uses():
                if u.user not in node_to_idx:
                    return
                idx = node_to_idx[u.user]
                end = max(end, idx)

            ptr_addr = re.sub(f" #.*\n", "", str(node))
            ptr_addr = re.sub(f", scope:.*", "", ptr_addr)
            ptr_addr = re.sub(":.*=", "=", ptr_addr)
            req_allocs.append(
                RequiredAlloc(
                    LiveRange(start, end),
                    int(np.prod(sizes)) * 4,
                    ptr_addr
                )
            )
        else:
            print(node, model._get_name())
            raise Exception

    with torch.jit._hide_source_ranges():
        node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
        for node in graph.nodes():
            make_alloc(node)
           
    mem_events = []
    for lvr in req_allocs:
        mem_events.append(MemEvent(lvr.ptr_addr, lvr.size, lvr.lvr.begin))
        mem_events.append(MemEvent(lvr.ptr_addr, -lvr.size, lvr.lvr.end + 1))
    return req_allocs, mem_events


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
    for mem_ev in get_mem_events(trace_json):
        ptr_addr, size, ts = mem_ev.ptr_addr, mem_ev.size, mem_ev.ts
        mem_events_paired[ptr_addr].append((size, ts))

    live_ranges = []
    for ptr_addr, mem_evts in mem_events_paired.items():
        if len(mem_evts) < 2:
            print("no free", mem_evts, file=sys.stderr)
            continue
        alloc_evt, free_evt = mem_evts
        alloc_size, alloc_ts = alloc_evt
        free_size, free_ts = free_evt
        assert alloc_size == -free_size
        assert alloc_ts < free_ts
        live_ranges.append(
            RequiredAlloc(LiveRange(alloc_ts, free_ts), alloc_size, ptr_addr)
        )
    return live_ranges


def dump_req_mem_allocs(reqs: List[RequiredAlloc], name):
    with open(f"req_allocs/{name}.txt", "w") as f:
        f.write(str(reqs).replace("[", "{").replace("]", "}"))


if __name__ == "__main__":
    sh_obj = ctypes.cdll.LoadLibrary("/home/mlevental/dev_projects/pytorch_memory_planning/cpp_src/cmake-build-debug/runtime_patch/libruntime_patch.so")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("name", type=str)
    args = parser.parse_args()
    name = f"{args.name}"
    model = vision_models(name)

    x = torch.rand((1, 3, 244, 244))
    # y = torch.jit.trace(model, (x,), strict=False)
    # y.save(f"models/{name}.pt")
    profile_model(model, x, name)
    # req_mem_allocs, mem_events = get_reqs_from_trace(name)
    # req_mem_allocs, mem_events = analyze_model(model, x)
    # dump_req_mem_allocs(req_mem_allocs, name)
