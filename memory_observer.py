import json
import re
from collections import defaultdict, namedtuple
from pprint import pprint

import torch
import yaml
from IPython.core.display import display
from torch import TensorType, OptionalType, IntType
from torch._C._autograd import DeviceType


def is_tensor_typ(typ):
    is_tensor = isinstance(typ, TensorType)
    is_opt = isinstance(typ, OptionalType)
    is_opt_tensor = is_opt and isinstance(typ.getElementType(), TensorType)
    return is_tensor or (is_opt and is_opt_tensor)


def parse_ops_yaml(fp, model_name):
    with open(fp, "r") as stream:
        ops_dict = yaml.safe_load(stream)

    allocations = {}
    addr_to_alloc = {}
    for alloc in ops_dict["allocations_list"]:
        if alloc["fn_name"] == "allocate":
            name = f"$alloc_{alloc['op_id']}"
            assert name not in allocations
            allocations[name] = alloc['addr']
            assert alloc['addr'] not in addr_to_alloc
            addr_to_alloc[alloc['addr']] = name

    ops_dict["addr_to_alloc"] = addr_to_alloc
    ops_dict["allocations"] = allocations

    # ops_dict_txt = open(fp, "r").read()
    # empty_tensors = re.findall(r"empty_tensor_\d*", ops_dict_txt)
    #
    # min_addr = float("inf")
    # max_addr = 0
    # for e in empty_tensors:
    #     min_addr = min(min_addr, int(e.replace("empty_tensor_", "")))
    #     max_addr = max(max_addr, int(e.replace("empty_tensor_", "")))
    #
    # max_size = 0
    # ptr_name_ptr_addr = ops_dict['constants']
    # alloc_addr_to_alloc = defaultdict(list)
    # for alloc in ops_dict["allocations"]:
    #     if alloc["fn_name"] == "allocate":
    #         alloc_addr_to_alloc[alloc["addr"]].append(alloc)
    #         min_addr = min(min_addr, alloc["addr"])
    #         max_addr = max(max_addr, alloc["addr"] + alloc["size"])
    #         max_size = max(max_size, alloc["size"])
    #
    # new_offset = max_addr + max_size
    # for addr, allocs in alloc_addr_to_alloc.items():
    #     new_offset += max_size
    #     for alloc in allocs:
    #         new_offset += max_size
    #         ptr_name_ptr_addr[f"$alloc_{alloc['op_id']}"] = new_offset + alloc["addr"]
    #
    # alloc_stack = []
    # remapped_frees = {}
    # for alloc in ops_dict["allocations"]:
    #     if alloc["fn_name"] == "allocate":
    #         alloc_stack.append(alloc)
    #     else:
    #         alloc_idx, alloc_alloc = next(
    #             (i, a) for i, a in enumerate(reversed(alloc_stack)) if a["addr"] == alloc["addr"])
    #         assert alloc["size"] == alloc_alloc["size"]
    #         remapped_frees[alloc["op_id"]] = ptr_name_ptr_addr[f"$alloc_{alloc_alloc['op_id']}"]
    #
    # empty_tensors = 0

    id_to_op = {}

    def loop(vals, typs, schema_args=None):
        for i, (val, arg_type) in enumerate(zip(vals, typs)):
            if schema_args is not None:
                # assert arg_type == schema_args[i].type.annotation_str
                typs[i] = schema_args[i].type
            if val == "None":
                assert isinstance(typs[i], OptionalType)
                vals[i] = None
            if isinstance(arg_type, str) and arg_type == "Device":
                vals[i] = getattr(DeviceType, val.upper())

    def fill_in_type(op):
        id_to_op[op["op_id"]] = op

        if op["fn_name"] in {"allocate", "free"}:
            return
        schema = namedtuple('schema', ['arguments', 'returns'])(None, None)
        if op["schema"] != "no schema":
            schema = torch._C.parse_schema(op["schema"])
        else:
            # TODO: top-level -> ? maybe do something with return type of graph?
            for i, r in enumerate(op["args types"]):
                if r == "Tensor":
                    op["args types"][i] = TensorType.get()
            for i, r in enumerate(op["returns types"]):
                if r == "Tensor":
                    op["returns types"][i] = TensorType.get()

        loop(op["args"], op["args types"], schema.arguments)
        loop(op["returns"], op["returns types"], schema.returns)

        if "calls" in op and op["calls"] is None:
            op["calls"] = []
        for o in op["calls"]:
            fill_in_type(o)

    for op in ops_dict["trace"]:
        fill_in_type(op)

    # for op_id, op in id_to_op.items():
    #     if op["fn_name"] in {"allocate", "free"}:
    #         if op["fn_name"] == "allocate":
    #             assert f"$alloc_{op['op_id']}" in ptr_name_ptr_addr
    #             op["addr"] = ptr_name_ptr_addr[f"$alloc_{op['op_id']}"]
    #         else:
    #             op["addr"] = remapped_frees[op['op_id']]
    #     # if is_tensor_typ(typs[i]) and isinstance(val, str) and val.startswith("empty_tensor_"):
    #     #     val = int(val.replace("empty_tensor_", ""))
    #     #     vals[i] = val
    #     #     ptr_name_ptr_addr[f"$empty_tensor_{empty_tensors}"] = val
    #     #     empty_tensors += 1

    return ops_dict, id_to_op


def find_out_variants(ops_dict, model_name):
    out_variants = {}

    def check_schema(op):
        schemas = []
        if op["fn_name"] not in {"allocate", "free", model_name}:
            schemas = torch._C._jit_get_schemas_for_operator(op["fn_name"])
        for s in schemas:
            if "out" in s.overload_name:
                out_variants[op["fn_name"]] = s
        for o in (op.get("calls") or []):
            check_schema(o)

    for op in ops_dict["trace"]:
        check_schema(op)

    return out_variants


def label_pointers(ops_dict, id_to_op, model_name):
    ops_dict = dict(ops_dict)
    ptr_name_ptr_addr = ops_dict['constants']
    ptr_addr_to_name = {}
    for k, v in ptr_name_ptr_addr.items():
        # TODO: need to handle actual constants here
        if isinstance(v, str):
            ptr_addr_to_name[v] = k
    addr_to_alloc = ops_dict['addr_to_alloc']
    for addr, alloc_name in addr_to_alloc.items():
        ptr_addr_to_name[f"tensor_ptr_{addr}"] = alloc_name

    graph_top_trace = next(t for t in ops_dict["trace"] if t["fn_name"] == model_name)
    graph_args = ops_dict["graph"]["$args"]
    graph_top_trace["args names"] = list(graph_args.keys())
    # TODO: handle graph return here
    graph_top_trace["returns names"] = []

    new_constants = {}

    def percolate(caller, rev=False):
        if "calls" not in caller: return
        args_or_rets = "returns" if rev else "args"
        parent_args_or_rets, parent_args_or_rets_names = caller[args_or_rets], caller[f"{args_or_rets} names"]
        if rev:
            calls = reversed(list(enumerate(caller["calls"])))
        else:
            calls = enumerate(caller["calls"])

        for call_idx, call in calls:
            if call["fn_name"] == "allocate":
                call["args names"] = ["nbytes"]
                call["args types"] = [IntType.get()]
                call["args"] = [call["size"]]
                call["returns names"] = [addr_to_alloc[call["addr"]]]
                call["returns types"] = [TensorType.get()]
                call["returns"] = [f"tensor_ptr_{call['addr']}"]
                continue
            if call["fn_name"] == "free":
                call["args names"] = ["ptr"]
                call["args types"] = [TensorType.get()]
                call["args"] = [f"tensor_ptr_{call['addr']}"]
                call["returns names"] = []
                call["returns types"] = []
                call["returns"] = []
                continue
            if f"{args_or_rets} names" not in call:
                call[f"{args_or_rets} names"] = [None] * len(call[args_or_rets])
                for arg_idx, (arg, typ) in enumerate(zip(call[args_or_rets], call[f"{args_or_rets} types"])):
                    typ = call[f"{args_or_rets} types"][arg_idx]
                    if isinstance(typ, TensorType):
                        if not any(i for i, p_arg in enumerate(parent_args_or_rets) if arg == p_arg):
                            if "empty" in arg: continue
                            assert arg in ptr_addr_to_name
                            arg_name = ptr_addr_to_name[arg]
                        else:
                            parent_args_idx = next(i for i, p_arg in enumerate(parent_args_or_rets) if arg == p_arg)
                            arg_name = parent_args_or_rets_names[parent_args_idx]
                    else:
                        arg_name = f"new_constants_{len(new_constants)}"
                        new_constants[arg_name] = arg
                    call[f"{args_or_rets} names"][arg_idx] = arg_name
            percolate(call, rev)

    percolate(graph_top_trace)
    percolate(graph_top_trace, rev=True)

    ops_dict["graph_top_trace"] = graph_top_trace
    return ops_dict

def print_ops_dict(ops_dict, name):
    graph_top_trace = dict(ops_dict["graph_top_trace"])

    def loop(vals, typs):
        for i in range(len(vals)):
            typs[i] = str(typs[i].annotation_str)
            if "Device" in typs[i] and vals[i] is not None:
                vals[i] = vals[i].name

    def stringify_type(op):
        loop(op["args"], op["args types"])
        loop(op["returns"], op["returns types"])

        if op.get("calls") is None:
            op["calls"] = []
        for o in op["calls"]:
            stringify_type(o)

    stringify_type(graph_top_trace)
    yaml.preserve_quotes = False
    with open(f"{name}_ops_dict.yml", "w") as f:
        yaml.dump(graph_top_trace, f, default_flow_style=None, width=1000, sort_keys=True)
    with open(f"{name}_ops_dict.txt", "w") as f:
        pprint(graph_top_trace, f, width=200, sort_dicts=False, indent=1)
    with open(f"{name}_ops_dict.json", "w") as f:
        json.dump(graph_top_trace, f, indent=2)


def find_final_dispatch(ops_dict):
    torch._C.parse_schema(
        "aten::_slow_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput)")
    torch._C._jit_get_operation("aten::_slow_conv2d_forward")
    torch._C._jit_get_schemas_for_operator("aten::_slow_conv2d_forward")


if __name__ == "__main__":
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
    # ops_dict = parse_ops_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/memory_allocator/ops.bkup.yml")
    ops_dict, id_to_op = parse_ops_yaml(
        "/Users/mlevental/dev_projects/pytorch_shape_inference/memory_allocator/ops.yml",
        "resnet18")
    ops_dict = label_pointers(ops_dict, id_to_op, "resnet18")
    print_ops_dict(ops_dict, "resnet18")
    # find_out_variants(ops_dict, "resnet18")
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
