import glob
import sys
from collections import defaultdict, namedtuple
from pprint import pprint

import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.width = 4096
yaml.indent = 4

import torch
from torch import TensorType, OptionalType, IntType, ListType, DictType, StringType
from torch._C._autograd import DeviceType


# def is_tensor_typ(typ):
#     is_tensor = isinstance(typ, TensorType)
#     is_opt = isinstance(typ, OptionalType)
#     is_opt_tensor = is_opt and isinstance(typ.getElementType(), TensorType)
#     return is_tensor or (is_opt and is_opt_tensor)

def find_graph_top(ops_dict, model_name):
    return next(t for t in ops_dict["trace"] if t["fn_name"] == model_name)


def parse_ops_yaml(fp, model_name):
    import yaml

    with open(fp, "r") as stream:
        ops_dict = yaml.load(stream, Loader=yaml.CLoader)

    name_order = ops_dict["name_order"] = {
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

    id_to_op = {}
    ptr_addr_to_name = {}

    def loop(vals, typs, names, schema_args=None):
        nonlocal ptr_addr_to_name
        for i in range(len(vals)):
            if schema_args is not None:
                typs[i] = schema_args[i].type
            if isinstance(typs[i], TensorType) or isinstance(typs[i], str) and typs[i] == "Tensor":
                typs[i] = TensorType.get()
                if i < len(names):
                    if vals[i] in ptr_addr_to_name:
                        if ptr_addr_to_name[vals[i]] != names[i]:
                            assert name_order[ptr_addr_to_name[vals[i]]] < name_order[names[i]]
                    else:
                        ptr_addr_to_name[vals[i]] = names[i]
            if isinstance(typs[i], str) and "String" in typs[i]:
                typs[i] = StringType.get()
            if isinstance(typs[i], str) and "Tensor" in typs[i]:
                typs[i] = TensorType.get()
            if isinstance(typs[i], str) and "List" in typs[i]:
                typs[i] = ListType(IntType.get())
            if isinstance(typs[i], dict):
                k, v = typs[i]["Dict"]
                if isinstance(k, StringType) and isinstance(v, TensorType):
                    typs[i] = DictType(k, v)
                else:
                    raise "wtf"

            if vals[i] == "None":
                assert isinstance(typs[i], OptionalType)
                vals[i] = None
            if isinstance(typs[i], str) and typs[i] == "Device":
                vals[i] = getattr(DeviceType, vals[i].upper())

    def fill_in_type(op):
        nonlocal ops_dict, ptr_addr_to_name
        id_to_op[op["op_id"]] = op

        if op["fn_name"] == "allocate":
            op["args names"] = ["nbytes"]
            op["args types"] = [IntType.get()]
            op["args"] = [op["size"]]
            op["returns names"] = [addr_to_alloc[op["addr"]]]
            op["returns types"] = [TensorType.get()]
            op["returns"] = [f"tensor_ptr_{op['addr']}"]
            op["calls"] = []
            return
        if op["fn_name"] == "free":
            op["args names"] = [addr_to_alloc[op["addr"]]]
            op["args types"] = [TensorType.get()]
            op["args"] = [f"tensor_ptr_{op['addr']}"]
            op["returns names"] = []
            op["returns types"] = []
            op["returns"] = []
            op["calls"] = []
            return

        schema = namedtuple("schema", ["arguments", "returns"])(None, None)
        if op["schema"] != "no schema":
            schema = torch._C.parse_schema(op["schema"])
        else:
            # graph inputs
            if op["fn_name"] == model_name:
                op["args names"] = list(ops_dict["graph"]["$args"].keys())
                op["args types"] = list(ops_dict["graph"]["$args"].values())
                op["returns names"] = ops_dict["graph"]["return"].replace("(", "").replace(")", "").split(",")
            elif op["fn_name"] == "prim::DictConstruct":
                op["returns types"] = [{"Dict": op["args types"]}]
                op["returns"] = [{}]
                ptr_addr_to_name[tuple(op["args"])] = op["returns names"][0]
                for i in range(0, len(op["args"]), 2):
                    op["returns"][0][op["args"][i]] = op["args"][i + 1]
            elif op["fn_name"] == "prim::ListConstruct":
                op["returns types"] = ["List"]
                op["returns"] = [op["args"]]
                ptr_addr_to_name[tuple(op["args"])] = op["returns names"][0]

        loop(op["args"], op["args types"], op["args names"], schema.arguments)
        loop(op["returns"], op["returns types"], op["returns names"], schema.returns)

        if "calls" in op and op["calls"] is None:
            op["calls"] = []
        for o in op["calls"]:
            o["caller_op_id"] = op["op_id"]
            if o.get("args names") is None:
                o["args names"] = []
            if o.get("returns names") is None:
                o["returns names"] = []
            fill_in_type(o)

    for op in ops_dict["trace"]:
        if op.get("args names") is None:
            op["args names"] = []
        if op.get("returns names") is None:
            op["returns names"] = []
        fill_in_type(op)

    for k, v in ops_dict["constants"].items():
        # TODO: need to handle actual constants here
        if isinstance(v, str) and "tensor" in v:
            ptr_addr_to_name[v] = k
    # for addr, alloc_name in addr_to_alloc.items():
    #     ptr_addr_to_name[f"tensor_ptr_{addr}"] = alloc_name

    ops_dict["ptr_addr_to_name"] = ptr_addr_to_name
    ops_dict["outs"] = {
        name: dump
        for name, dump in ops_dict["graph"].items()
        if isinstance(dump, str) and "Constant" not in dump
    }
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
    addr_to_alloc = ops_dict["addr_to_alloc"]
    constants = ops_dict["constants"]
    name_order = ops_dict["name_order"]
    constants_to_names = {str(v): k for k, v in constants.items()}

    new_constant = 0
    reused_ptrs = defaultdict(list)

    known_aliases = set()

    def handle_tensor(call, arg_or_ret, parent_args_or_rets, parent_args_or_rets_names):
        nonlocal known_aliases, ptr_addr_to_name
        # get names from "parent" (caller)
        if not any(
                (i, p_arg_or_ret)
                for i, p_arg_or_ret in enumerate(parent_args_or_rets)
                if arg_or_ret == p_arg_or_ret
        ):
            # if not passed in from parent (and not empty) then it should be a known pointer
            if "empty" in arg_or_ret:
                return
            if arg_or_ret in ptr_addr_to_name:
                arg_or_ret_name = ptr_addr_to_name[arg_or_ret]
            elif arg_or_ret.replace("tensor_ptr_", "") in addr_to_alloc:
                arg_or_ret_name = addr_to_alloc[arg_or_ret.replace("tensor_ptr_", "")]
            else:
                raise "wtf"
        else:
            parent_args_idx = next(
                i
                for i, p_arg_or_ret in enumerate(parent_args_or_rets)
                if arg_or_ret == p_arg_or_ret
            )
            arg_or_ret_name = parent_args_or_rets_names[parent_args_idx]

        # update ptr to name mapping if we have new info
        if arg_or_ret not in ptr_addr_to_name:
            ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
        elif not (ptr_addr_to_name[arg_or_ret] == arg_or_ret_name):
            # valid reasons for repeats

            # updating with top level visible names
            if "alloc" in ptr_addr_to_name[arg_or_ret] and "alloc" not in arg_or_ret_name:
                ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
            # updating with dominating name (passed in and out of a single op y = f(X), then y aliases x
            elif (
                    name_order[arg_or_ret_name]
                    < name_order[ptr_addr_to_name[arg_or_ret]]
            ):
                # can't happen with allocs because they come out of `allocate`
                assert ("alloc" not in ptr_addr_to_name[arg_or_ret]) and (
                        "alloc" not in arg_or_ret_name
                )
                assert (ptr_addr_to_name[arg_or_ret], arg_or_ret_name) not in known_aliases
                # tuple order corresponds to name_order
                known_aliases.add((arg_or_ret_name, ptr_addr_to_name[arg_or_ret]))
                ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
            # we should be catching aliases in the correct order (later gets seen first because we percolate backwards first)
            elif (name_order[arg_or_ret_name]
                  >= name_order[ptr_addr_to_name[arg_or_ret]]):
                assert (ptr_addr_to_name[arg_or_ret], arg_or_ret_name) in known_aliases
            else:
                raise "wtf"

        return arg_or_ret_name

    def percolate(this_calls, parent_args_or_rets, parent_args_or_rets_names, rev):
        nonlocal new_constant, constants, constants_to_names

        if rev:
            this_calls = list(reversed(this_calls))

        args_or_rets = "returns" if rev else "args"
        for call in this_calls:
            names = [None] * len(call[args_or_rets])
            for arg_or_ret_idx, (arg_or_ret, typ) in enumerate(
                    zip(call[args_or_rets], call[f"{args_or_rets} types"])
            ):
                # TODO: optionals
                if isinstance(typ, ListType) and isinstance(
                        typ.getElementType(), TensorType
                ):
                    arg_or_retss = arg_or_ret
                    arg_or_ret_name = []
                    for arg_or_ret in arg_or_retss:
                        arg_or_ret_name.append(
                            handle_tensor(
                                call,
                                arg_or_ret,
                                parent_args_or_rets,
                                parent_args_or_rets_names,
                            )
                        )
                elif isinstance(typ, TensorType):
                    arg_or_ret_name = handle_tensor(
                        call,
                        arg_or_ret,
                        parent_args_or_rets,
                        parent_args_or_rets_names,
                    )
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

            this_args_or_rets = call[args_or_rets]
            this_args_or_rets_names = names
            if this_args_or_rets and not this_args_or_rets_names:
                raise "wtf"
            if this_args_or_rets and this_args_or_rets_names:
                this_calls = call["calls"]
                percolate(this_calls, this_args_or_rets, this_args_or_rets_names, rev)

    graph_top_trace = find_graph_top(ops_dict, model_name)
    for i, arg in enumerate(graph_top_trace["args"]):
        ptr_addr_to_name[arg] = graph_top_trace["args names"][i]

    calls = graph_top_trace["calls"]
    args = graph_top_trace["args"]
    args_names = graph_top_trace["args names"]
    percolate(calls, args, args_names, rev=False)

    returns = graph_top_trace["returns"]
    returns_names = graph_top_trace["returns names"]
    percolate(calls, returns, returns_names, rev=True)
    percolate(calls, args, args_names, rev=False)
    percolate(calls, returns, returns_names, rev=True)
    percolate(calls, args, args_names, rev=False)

    name_to_ptr = {v: k for k, v in ptr_addr_to_name.items()}
    graph_top_trace["returns"] = []
    for r in graph_top_trace["returns names"]:
        assert r in ops_dict["outs"]
        graph_top_trace["returns"].append(name_to_ptr[r])

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
    constants = ops_dict["constants"]
    alloc_ptr_to_alloc = {
        f"tensor_ptr_{a['addr']}": id_to_op[a["op_id"]]
        for a in ops_dict["allocations_list"]
        if a["fn_name"] == "allocate"
    }
    # outs = ops_dict["outs"]
    allocs = {}
    for ptr, fn_name in ptr_addr_to_name.items():
        if alloc := alloc_ptr_to_alloc.get(ptr):
            allocs[fn_name] = alloc

    call_chains = defaultdict(list)
    for fn_name, call in allocs.items():
        size = call["size"]
        while caller_op_id := call.get("caller_op_id"):
            call_chains[fn_name].append(caller_op_id)
            call = id_to_op[caller_op_id]

        qualified_call_chain = []
        for op_id in reversed(call_chains[fn_name]):
            if id_to_op[op_id]["schema"] != "no schema":
                fn = f"{id_to_op[op_id]['fn_name']}({{}})"
                args = []
                for arg in id_to_op[op_id]['args']:
                    if not isinstance(arg, str):
                        args.append(str(arg))
                    elif "tensor_ptr" in arg:
                        if arg in ptr_addr_to_name:
                            name = ptr_addr_to_name[arg]
                            if name in allocs:
                                args.append(
                                    # "ptr_" + str(allocs[name]['size'])
                                    name
                                )
                            elif fn_name in constants:
                                args.append(name)
                        else:
                            assert "empty_tensor" in arg
                            args.append(arg)
                    else:
                        args.append(arg)
                qualified_call_chain.append(
                    fn.format(", ".join(args))
                )
        qualified_call_chain.append(f"allocate({size})")
        call_chains[fn_name] = qualified_call_chain

    assert len(call_chains) == len(allocs)
    assert set(call_chains.keys()) == set(allocs.keys())
    unique_call_chains = defaultdict(list)
    for k, v in call_chains.items():
        unique_call_chains[tuple(v)].append(k)
    if len(unique_call_chains) != len(allocs):
        for c, ks in {c: ks for c, ks in unique_call_chains.items() if len(ks) > 1}.items():
            print(c, ks)
            for k in ks:
                pprint((k, allocs[k]["addr"], allocs[k]["size"]))

    return dict(call_chains), allocs


def print_call_chains(call_chains, output_allocs, file_name):
    # for name, call_chain in call_chains.items():
    #     call_chains[name] = flist(call_chain)
    for name, output_alloc in output_allocs.items():
        output_allocs[name] = stringify_type(output_alloc)

    with open(f"call_chains/{file_name}.yml", "w") as f:
        yaml.dump({"call_chains": call_chains, "output_allocs": output_allocs}, f)


def find_final_dispatch(ops_dict):
    torch._C.parse_schema(
        "aten::_slow_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput)"
    )
    torch._C._jit_get_operation("aten::_slow_conv2d_forward")
    torch._C._jit_get_schemas_for_operator("aten::_slow_conv2d_forward")


def list_all_ops(profiles_fp):
    import yaml

    ops = set()
    for fp in glob.glob(f"{profiles_fp}/*.yml"):
        try:
            with open(fp, "r") as stream:
                ops_dict = yaml.load(stream, Loader=yaml.CLoader)
            print(fp)
            for op_name, op in ops_dict["graph"].items():
                if op_name in {"$args", "return"}:
                    continue
                if len(op.split(" = ")) != 2:
                    print(op)
                _, op_sig = op.split(" = ")
                op, _args = op_sig.split("(")
                ops.add(op)
        except:
            pass
    pprint(ops)


if __name__ == "__main__":
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
    # ops_dict = parse_ops_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/memory_allocator/ops.bkup.yml")
    # model_name = "deeplabv3_mobilenet_v3_large"
    model_name = "efficientnet_b0"
    ops_dict = parse_ops_yaml(
        f"/home/mlevental/dev_projects/pytorch_memory_planning/profiles/{model_name}.yml",
        model_name
    )
    ops_dict = label_pointers(ops_dict, model_name)
    call_chains, output_allocs = match_allocs_to_outs(ops_dict)
    print_call_chains(call_chains, output_allocs, model_name)
    # print_ops_dict(ops_dict, "resnet18")
    # find_out_variants(ops_dict, "resnet18")
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
