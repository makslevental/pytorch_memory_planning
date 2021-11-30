import sys
from collections import defaultdict, namedtuple

import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.width = 4096
yaml.indent = 4

import torch
from torch import TensorType, OptionalType, IntType, ListType
from torch._C._autograd import DeviceType


# def is_tensor_typ(typ):
#     is_tensor = isinstance(typ, TensorType)
#     is_opt = isinstance(typ, OptionalType)
#     is_opt_tensor = is_opt and isinstance(typ.getElementType(), TensorType)
#     return is_tensor or (is_opt and is_opt_tensor)


def parse_ops_yaml(fp):
    import yaml
    with open(fp, "r") as stream:
        ops_dict = yaml.load(stream)

    ops_dict["name_order"] = {
        name: i for i, name in enumerate(ops_dict["graph"].keys())
    }
    allocations = {}
    addr_to_alloc = {}
    for alloc in ops_dict["allocations_list"]:
        if alloc["fn_name"] == "allocate":
            name = f"$alloc_{alloc['op_id']}"
            assert name not in allocations
            allocations[name] = alloc["addr"]
            assert alloc["addr"] not in addr_to_alloc
            addr_to_alloc[alloc["addr"]] = name

    ops_dict["addr_to_alloc"] = addr_to_alloc
    ops_dict["allocations"] = allocations

    ptr_name_ptr_addr = ops_dict["constants"]
    ptr_addr_to_name = {}
    for k, v in ptr_name_ptr_addr.items():
        # TODO: need to handle actual constants here
        if isinstance(v, str) and "tensor" in v:
            ptr_addr_to_name[v] = k
    for addr, alloc_name in addr_to_alloc.items():
        ptr_addr_to_name[f"tensor_ptr_{addr}"] = alloc_name

    ops_dict["ptr_addr_to_name"] = ptr_addr_to_name
    ops_dict["outs"] = {
        name: dump
        for name, dump in ops_dict["graph"].items()
        if isinstance(dump, str) and "Constant" not in dump
    }

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

        if op["fn_name"] == "allocate":
            op["args names"] = ["nbytes"]
            op["args types"] = [IntType.get()]
            op["args"] = [op["size"]]
            op["returns names"] = [addr_to_alloc[op["addr"]]]
            op["returns types"] = [TensorType.get()]
            op["returns"] = [f"tensor_ptr_{op['addr']}"]
            return
        if op["fn_name"] == "free":
            op["args names"] = [addr_to_alloc[op["addr"]]]
            op["args types"] = [TensorType.get()]
            op["args"] = [f"tensor_ptr_{op['addr']}"]
            op["returns names"] = []
            op["returns types"] = []
            op["returns"] = []
            return

        schema = namedtuple("schema", ["arguments", "returns"])(None, None)
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
            o["caller_op_id"] = op["op_id"]
            fill_in_type(o)

    for op in ops_dict["trace"]:
        fill_in_type(op)

    ops_dict["id_to_op"] = id_to_op

    return ops_dict


def find_out_variants(ops_dict, model_name):
    out_variants = {}

    def check_schema(op):
        schemas = []
        if op["fn_name"] not in {"allocate", "free", model_name}:
            schemas = torch._C._jit_get_schemas_for_operator(op["fn_name"])
        for s in schemas:
            if "out" in s.overload_name:
                out_variants[op["fn_name"]] = s
        for o in op.get("calls") or []:
            check_schema(o)

    for op in ops_dict["trace"]:
        check_schema(op)

    return out_variants




def label_pointers(ops_dict, model_name):
    ops_dict = dict(ops_dict)
    ptr_addr_to_name = ops_dict["ptr_addr_to_name"]
    constants = ops_dict["constants"]
    name_order = ops_dict["name_order"]
    constants_to_names = {str(v): k for k, v in constants.items()}

    graph_top_trace = next(t for t in ops_dict["trace"] if t["fn_name"] == model_name)
    graph_args = ops_dict["graph"]["$args"]
    graph_top_trace["args names"] = list(graph_args.keys())
    # TODO: handle graph return here
    graph_top_trace["returns names"] = []
    new_constant = 0
    reused_ptrs = defaultdict(list)

    def handle_tensor(call, arg_or_ret, parent_args_or_rets, parent_args_or_rets_names):
        # have to handle list args
        if not any(
                (i, p_arg_or_ret)
                for i, p_arg_or_ret in enumerate(parent_args_or_rets)
                if arg_or_ret == p_arg_or_ret
        ):
            # if not passed in from parent (and not empty) then it should be a known pointer
            if "empty" in arg_or_ret:
                return
            assert arg_or_ret in ptr_addr_to_name
            arg_or_ret_name = ptr_addr_to_name[arg_or_ret]
        else:
            parent_args_idx = next(
                i
                for i, p_arg_or_ret in enumerate(parent_args_or_rets)
                if arg_or_ret == p_arg_or_ret
            )
            arg_or_ret_name = parent_args_or_rets_names[parent_args_idx]
            if arg_or_ret in ptr_addr_to_name:
                if (
                        ptr_addr_to_name[arg_or_ret] == arg_or_ret_name
                        or (
                        "alloc" in ptr_addr_to_name[arg_or_ret]
                        and "alloc" not in arg_or_ret_name
                )
                        or "empty_tensor" in arg_or_ret
                ):
                    ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
                elif (
                        name_order[arg_or_ret_name]
                        < name_order[ptr_addr_to_name[arg_or_ret]]
                ):
                    assert (
                                   "alloc" not in ptr_addr_to_name[arg_or_ret]
                           ) and ("alloc" not in arg_or_ret_name)
                    ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
                else:
                    reused_ptrs[arg_or_ret].append(
                        (
                            call["op_id"],
                            arg_or_ret_name,
                            ptr_addr_to_name[arg_or_ret],
                        )
                    )

        return arg_or_ret_name

    def percolate(caller, rev=False):
        nonlocal new_constant

        if "calls" not in caller:
            return
        if rev:
            calls = list(reversed(caller["calls"]))
        else:
            calls = list(caller["calls"])

        args_or_rets = "returns" if rev else "args"
        parent_args_or_rets, parent_args_or_rets_names = (
            caller[args_or_rets],
            caller[f"{args_or_rets} names"],
        )
        for call in calls:
            if parent_args_or_rets and parent_args_or_rets_names:
                names = [None] * len(call[args_or_rets])
                for arg_or_ret_idx, (arg_or_ret, typ) in enumerate(
                    zip(call[args_or_rets], call[f"{args_or_rets} types"])
                ):
                    typ = call[f"{args_or_rets} types"][arg_or_ret_idx]
                    if isinstance(typ, ListType) and isinstance(typ.getElementType(), TensorType):
                        arg_or_retss = arg_or_ret
                        arg_or_ret_name = []
                        for arg_or_ret in arg_or_retss:
                            arg_or_ret_name.append(handle_tensor(call, arg_or_ret, parent_args_or_rets, parent_args_or_rets_names))
                    elif isinstance(typ, TensorType):
                        arg_or_ret_name = handle_tensor(call, arg_or_ret, parent_args_or_rets, parent_args_or_rets_names)
                    elif str(arg_or_ret) in constants_to_names:
                        arg_or_ret_name = constants_to_names[str(arg_or_ret)]
                    else:
                        print(call["op_id"], arg_or_ret, file=sys.stderr)
                        new_constant += 1
                        arg_or_ret_name = f"$new_constant_{new_constant}"
                        constants[arg_or_ret_name] = str(arg_or_ret)
                        constants_to_names[str(arg_or_ret)] = arg_or_ret_name

                    names[arg_or_ret_idx] = arg_or_ret_name
                if f"{args_or_rets} names" in call:
                    assert len(list(filter(None, names))) >= len(
                        list(filter(None, call[f"{args_or_rets} names"]))
                    )
                call[f"{args_or_rets} names"] = names
            percolate(call, rev)

    assert not reused_ptrs
    percolate(graph_top_trace, rev=True)
    percolate(graph_top_trace)

    ops_dict["graph_top_trace"] = graph_top_trace
    return ops_dict


def flist(x):
    retval = ruamel.yaml.comments.CommentedSeq(x)
    retval.fa.set_flow_style()  # fa -> format attribute
    return retval


def flist_op(op):
    op["args"] = flist(op["args"])
    op["args types"] = flist(op["args types"])
    op["returns"] = flist(op["returns"])
    op["returns types"] = flist(op["returns types"])
    if "args names" in op:
        op["args names"] = flist(op["args names"])
    if "returns names":
        op["returns names"] = flist(op["returns names"])
    return op


def stringify_type(op):
    def loop(vals, typs, names):
        for i, t in enumerate(typs):
            typs[i] = str(typs[i].annotation_str)
        for i, v in enumerate(vals):
            if "Device" in typs[i] and v is not None:
                vals[i] = v.name
        for i, n in enumerate(names):
            if names[i] is not None:
                names[i] = names[i].replace("$", "%")

    def _stringify_type(op):
        del op["op_id"]
        if "caller_op_id" in op:
            del op["caller_op_id"]
        loop(op["args"], op["args types"], op.get("args names", []))
        loop(op["returns"], op["returns types"], op.get("returns names", []))
        op = flist_op(op)
        if op.get("calls") is None:
            op["calls"] = []
        for o in op["calls"]:
            _stringify_type(o)

    _stringify_type(op)
    return op


def print_ops_dict(ops_dict, name):
    graph_top_trace = dict(ops_dict["graph_top_trace"])
    stringify_type(graph_top_trace)

    with open(f"{name}_ops_dict.yml", "w") as f:
        # yaml.dump(graph_top_trace, f, default_flow_style=None, width=1000, sort_keys=False)
        yaml.dump(graph_top_trace, f)
    # with open(f"{name}_ops_dict.txt", "w") as f:
    #     pprint(graph_top_trace, f, width=200, sort_dicts=False, indent=1)
    # with open(f"{name}_ops_dict.json", "w") as f:
    #     json.dump(graph_top_trace, f, indent=2)


def match_allocs_to_outs(ops_dict):
    ptr_addr_to_name = ops_dict["ptr_addr_to_name"]
    id_to_op = ops_dict["id_to_op"]
    alloc_ptr_to_alloc = {
        f"tensor_ptr_{a['addr']}": id_to_op[a["op_id"]]
        for a in ops_dict["allocations_list"]
        if a["fn_name"] == "allocate"
    }
    # outs = ops_dict["outs"]
    allocs = {}
    for ptr, name in ptr_addr_to_name.items():
        if alloc := alloc_ptr_to_alloc.get(ptr):
            allocs[name] = alloc

    call_chains = defaultdict(list)
    for name, call in allocs.items():
        while caller_op_id := call.get("caller_op_id"):
            call_chains[name].append(caller_op_id)
            call = id_to_op[caller_op_id]

        qualified_call_chain = []
        for op_id in reversed(call_chains[name]):
            if id_to_op[op_id]["schema"] != "no schema":
                qualified_call_chain.append(
                    f"{id_to_op[op_id]['fn_name']}({', '.join(map(lambda x: str(x) if not isinstance(x, str) else ('ptr' if 'tensor' in x else x), id_to_op[op_id]['args']))})"
                )
        call_chains[name] = qualified_call_chain

    return dict(call_chains), allocs


def print_call_chains(call_chains, output_allocs, file_name):
    # for name, call_chain in call_chains.items():
    #     call_chains[name] = flist(call_chain)
    for name, output_alloc in output_allocs.items():
        output_allocs[name] = stringify_type(output_alloc)

    with open(f"{file_name}_call_chains.yml", "w") as f:
        yaml.dump({"call_chains": call_chains, "output_allocs": output_allocs}, f)


def find_final_dispatch(ops_dict):
    torch._C.parse_schema(
        "aten::_slow_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput)"
    )
    torch._C._jit_get_operation("aten::_slow_conv2d_forward")
    torch._C._jit_get_schemas_for_operator("aten::_slow_conv2d_forward")


if __name__ == "__main__":
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
    # ops_dict = parse_ops_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/memory_allocator/ops.bkup.yml")
    model_name = "unet"
    ops_dict = parse_ops_yaml(
        "/Users/mlevental/dev_projects/pytorch_shape_inference/memory_allocator/ops_unet_64.yml",
    )
    ops_dict = label_pointers(ops_dict, "unet")
    call_chains, output_allocs = match_allocs_to_outs(ops_dict)
    print_call_chains(call_chains, output_allocs, model_name)
    # print_ops_dict(ops_dict, "resnet18")
    # find_out_variants(ops_dict, "resnet18")
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
