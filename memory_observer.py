import glob
import json
import warnings
from collections import defaultdict, namedtuple
from pprint import pprint
from typing import Dict, Tuple

import ruamel.yaml

from strategies import RequiredAlloc, LiveRange

yaml = ruamel.yaml.YAML()
yaml.width = 4096
yaml.indent = 4

import torch
from torch import TensorType, IntType, ListType, DictType, StringType, BoolType
from torch._C._autograd import DeviceType


# def is_tensor_typ(typ):
#     is_tensor = isinstance(typ, TensorType)
#     is_opt = isinstance(typ, OptionalType)
#     is_opt_tensor = is_opt and isinstance(typ.getElementType(), TensorType)
#     return is_tensor or (is_opt and is_opt_tensor)


def find_idx(val, ls):
    return next(i for i, v in enumerate(ls) if v == val)


def find_graph_top(ops_dict, model_name):
    return next(t for t in ops_dict["calls"] if t["fn_name"] == model_name)


def load_ops_yaml(model_name):
    fix_yaml(model_name)
    import yaml

    with open(f"profiles/{model_name}.yml", "r") as stream:
        ops_dict = yaml.load(stream, Loader=yaml.CLoader)
    ops_dict["model_name"] = model_name
    return ops_dict


def ls_or_empty(op, key):
    return op.get(key, [])


def get_torch_type_of_python_val(val):
    if isinstance(val, int):
        return IntType.get()
    if isinstance(val, str):
        return StringType.get()
    if isinstance(val, bool):
        return BoolType.get()


def make_torch_type(typ, nested_type=None):
    if isinstance(typ, str) and "Int" in typ:
        typ = IntType.get()
    elif isinstance(typ, str) and "String" in typ:
        typ = StringType.get()
    elif isinstance(typ, str) and "Tensor" in typ:
        typ = TensorType.get()
    elif isinstance(typ, str) and "List" in typ:
        assert nested_type is not None
        typ = ListType(make_torch_type(nested_type))
    elif isinstance(typ, dict):
        [(k, v)] = list(typ.items())
        typ = DictType(make_torch_type(k), make_torch_type(v))
    else:
        raise "unrecognized typ"
    return typ


def parse_ops_dict(ops_dict):
    model_name = ops_dict["model_name"]
    id_to_op = {}

    def dfs(op):
        id_to_op[op["op_id"]] = op
        for o in op.get("calls") or []:
            dfs(o)

    for call in ops_dict["calls"]:
        dfs(call)

    allocations = {}
    addr_to_alloc = {}
    for alloc in ops_dict["allocations_list"]:
        if alloc["fn_name"] == "allocate":
            name = f"$alloc_{alloc['op_id']}"
            assert name not in allocations
            allocations[name] = alloc["addr"]
            assert alloc["addr"] not in addr_to_alloc
            addr_to_alloc[alloc["addr"]] = name

    def fill_args_types_names_op(op):
        for k in ["calls", "returns", "returns types", "returns names"]:
            if op.get(k) is None:
                op[k] = []
        if op["fn_name"] == "allocate":
            op["args names"] = ["nbytes"]
            op["args types"] = [IntType.get()]
            op["args"] = [op["size"]]
            op["returns names"] = [addr_to_alloc[op["addr"]]]
            op["returns types"] = [TensorType.get()]
            op["returns"] = [f"tensor_ptr_{op['addr']}"]
            return
        elif op["fn_name"] == "free":
            op["args names"] = [addr_to_alloc[op["addr"]]]
            op["args types"] = [TensorType.get()]
            op["args"] = [f"tensor_ptr_{op['addr']}"]
            return
        elif op["schema"] != "no schema":
            schema = torch._C.parse_schema(op["schema"])
            if "args types" not in op:
                raise "schema but no types?"
            types = op["args types"]
            for i, typ in enumerate(op["args types"]):
                types[i] = schema.arguments[i].type
            types = op["returns types"]
            for i, typ in enumerate(op["returns types"]):
                types[i] = schema.returns[i].type
        elif op["fn_name"] == "prim::DictConstruct":
            op["returns types"] = [make_torch_type(dict([op["args types"]]))]
            op["returns"] = [{}]
            for i in range(0, len(op["args"]), 2):
                op["returns"][0][op["args"][i]] = op["args"][i + 1]
            op["args types"] = [make_torch_type(a) for a in op["args types"]]
        elif op["fn_name"] == "prim::ListConstruct":
            op["returns types"] = [make_torch_type("List", op["args types"][0])]
            op["args types"] = [make_torch_type(a) for a in op["args types"]]
            op["returns"] = [op["args"]]
        else:
            pass
            # warnings.warn("no schema: " + op["fn_name"])

        for i, arg in enumerate(op["args"]):
            if isinstance(arg, list):
                op["args"][i] = tuple(arg)
        for i, arg in enumerate(op["returns"]):
            if isinstance(arg, list):
                op["returns"][i] = tuple(arg)

    for op in id_to_op.values():
        fill_args_types_names_op(op)

    name_order = {}
    i = 0
    for id, op in id_to_op.items():
        for name in ls_or_empty(op, "returns names"):
            i += 1
            name_order[name] = (i, id)

    graph_top = find_graph_top(ops_dict, model_name)
    graph_top["args types"] = [make_torch_type(typ) for typ in graph_top["args types"]]
    graph_top["args names"] = list(ops_dict["graph"]["$args"].keys())
    last_compute_op_id = next(
        compute_op_id
        for name, (_order, compute_op_id) in reversed(name_order.items())
        if "alloc" not in name
    )
    graph_top["returns"] = id_to_op[last_compute_op_id]["returns"]
    graph_top["returns types"] = id_to_op[last_compute_op_id]["returns types"]
    graph_top["returns names"] = id_to_op[last_compute_op_id]["returns names"]

    outs = []
    for call in graph_top["calls"]:
        outs.extend(call["returns names"])
    assert len(outs) == len(set(outs))

    for id, op in id_to_op.items():
        for o in op["calls"]:
            o["caller_op_id"] = op["op_id"]

    name_to_ptr = {
        name: arg for arg, name in zip(graph_top["args"], graph_top["args names"])
    }
    for id, op in id_to_op.items():
        for i, name in enumerate(ls_or_empty(op, "returns names")):
            if isinstance(op["returns types"][i], TensorType):
                if name in name_to_ptr:
                    assert name in graph_top["returns names"]
                else:
                    name_to_ptr[name] = op["returns"][i]
        for i, name in enumerate(ls_or_empty(op, "args names")):
            if isinstance(op["args types"][i], TensorType):
                ptr = op["args"][i]
                if name in name_to_ptr:
                    assert name_to_ptr[name] == ptr
                else:
                    name_to_ptr[name] = ptr

    for name, alloc_ptr in allocations.items():
        name_to_ptr[name] = f"tensor_ptr_{alloc_ptr}"

    for cons_name, maybe_ptr in ops_dict["constants"].items():
        if isinstance(maybe_ptr, str) and "tensor_ptr" in maybe_ptr:
            name_to_ptr[cons_name] = maybe_ptr

    ptr_to_names = defaultdict(list)
    for name, ptr in name_to_ptr.items():
        ptr_to_names[ptr].append(name)

    ptr_addr_to_name = {}
    for ptr, names in ptr_to_names.items():
        if len(names) > 1:
            # case 1 function call with output name for allocation
            if len(names) == 2:
                assert "alloc" not in names[0]
                assert "alloc" in names[1]
                assert name_order[names[0]] < name_order[names[1]]
            # case 2 after an alloc, true aliasing by passing through function boundary
            else:
                names = [
                    max(
                        [n for n in names if "alloc" not in n],
                        key=lambda x: name_order[x],
                    )
                ]
                # allocs = [(i, n) for i, n in enumerate(names) if "alloc" in n]
                # others = [(i, n) for i, n in enumerate(names) if "alloc" not in n]
                #
                #
                # for i, alloc_name in allocs:
                #     for _, other_name in filter(lambda x: x[0] < i, others):
                #         if name_order[other_name] > name_order[alloc_name]:
                #             warnings.warn(f"{other_name} {alloc_name}: {name_order[other_name]} < {name_order[alloc_name]}")
                #     for _, other_name in filter(lambda x: x[0] > i, others):
                #         warnings.warn(f"{alloc_name} {other_name}: {name_order[alloc_name]} > {name_order[other_name]}")
                #
                # for i in range(len(others) - 1):
                #     (_, return_name), (_, arg_name) = others[i], others[i + 1]
                #     assert name_order[return_name] < name_order[arg_name]
                #     # assert name_order[return_name] < name_order[alloc_name] < name_order[arg_name]
                #     first_op = id_to_op[name_order[return_name][1]]
                #     second_op = id_to_op[name_order[arg_name][1]]
                #     assert ptr in first_op["returns"]
                #     assert ptr in second_op["args"]
                names = [names[-1]]

        assert ptr not in ptr_addr_to_name
        ptr_addr_to_name[ptr] = names[0]

    ops_dict["outs"] = set(outs)
    ops_dict["id_to_op"] = id_to_op
    ops_dict["name_order"] = name_order
    ops_dict["name_to_ptr"] = name_to_ptr
    ops_dict["ptr_addr_to_name"] = ptr_addr_to_name
    return ops_dict


def find_out_variants(ops_dict):
    model_name = ops_dict["model_name"]
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

    for op in ops_dict["calls"]:
        check_schema(op)

    return out_variants


def label_pointers(ops_dict):
    model_name = ops_dict["model_name"]
    ops_dict = dict(ops_dict)
    ptr_addr_to_name = ops_dict["ptr_addr_to_name"]
    name_order = ops_dict["name_order"]

    new_constant = 0

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
            else:
                warnings.warn(
                    f"couldn't identify arg: {arg_or_ret} to call {call['op_id']}"
                )
                return None
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
            # updating with top level visible names
            if (
                "alloc" in ptr_addr_to_name[arg_or_ret]
                and "alloc" not in arg_or_ret_name
            ):
                ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
            # updating with dominating name (passed in and out of a single op y = f(X), then y aliases x
            elif name_order[arg_or_ret_name] < name_order[ptr_addr_to_name[arg_or_ret]]:
                # can't happen with allocs because they come out of `allocate`
                assert ("alloc" not in ptr_addr_to_name[arg_or_ret]) and (
                    "alloc" not in arg_or_ret_name
                )
                assert (
                    ptr_addr_to_name[arg_or_ret],
                    arg_or_ret_name,
                ) not in known_aliases
                # tuple order corresponds to name_order
                known_aliases.add((arg_or_ret_name, ptr_addr_to_name[arg_or_ret]))
                ptr_addr_to_name[arg_or_ret] = arg_or_ret_name
            # we should be catching aliases in the correct order (later gets seen first because we percolate backwards first)
            elif (
                name_order[arg_or_ret_name] >= name_order[ptr_addr_to_name[arg_or_ret]]
            ):
                assert (ptr_addr_to_name[arg_or_ret], arg_or_ret_name) in known_aliases
            else:
                raise Exception("wtf")

        return arg_or_ret_name

    constants = ops_dict["constants"]
    constants_to_names = {}
    for k, v in constants.items():
        if isinstance(v, list):
            v = tuple(v)
        constants[k] = v
        constants_to_names[v] = k

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
                    if f"{args_or_rets} names" in call:
                        arg_or_ret_name = call[f"{args_or_rets} names"][arg_or_ret_idx]
                    else:
                        arg_or_ret_name = handle_tensor(
                            call,
                            arg_or_ret,
                            parent_args_or_rets,
                            parent_args_or_rets_names,
                        )
                    # arg_or_retss = arg_or_ret
                    # arg_or_ret_name = []
                    # for arg_or_ret in arg_or_retss:
                    #     arg_or_ret_name.append(
                    #         handle_tensor(
                    #             call,
                    #             arg_or_ret,
                    #             parent_args_or_rets,
                    #             parent_args_or_rets_names,
                    #         )
                    #     )
                elif isinstance(typ, DictType):
                    arg_or_ret_name = parent_args_or_rets_names[0]
                elif isinstance(typ, TensorType):
                    arg_or_ret_name = handle_tensor(
                        call,
                        arg_or_ret,
                        parent_args_or_rets,
                        parent_args_or_rets_names,
                    )
                elif arg_or_ret in constants_to_names:
                    arg_or_ret_name = constants_to_names[arg_or_ret]
                else:
                    if isinstance(arg_or_ret, str) and "tensor_ptr" in arg_or_ret:
                        raise Exception("wtf")
                    # warnings.warn(f'new constant {call["op_id"]} {arg_or_ret}')
                    new_constant += 1
                    arg_or_ret_name = f"$new_constant_{new_constant}"
                    constants[arg_or_ret_name] = arg_or_ret
                    constants_to_names[arg_or_ret] = arg_or_ret_name

                names[arg_or_ret_idx] = arg_or_ret_name

            if f"{args_or_rets} names" in call:
                assert len(list(filter(None, names))) >= len(
                    list(filter(None, call[f"{args_or_rets} names"]))
                )
            call[f"{args_or_rets} names"] = names

            this_args_or_rets = call[args_or_rets]
            this_args_or_rets_names = names
            if this_args_or_rets and not this_args_or_rets_names:
                raise Exception("wtf")
            if this_args_or_rets and this_args_or_rets_names:
                this_calls = call["calls"]
                percolate(this_calls, this_args_or_rets, this_args_or_rets_names, rev)

    graph_top_trace = find_graph_top(ops_dict, model_name)
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

    # for k, v in constants.items():
    #     if isinstance(v, list):
    #         v = tuple(v)
    #     constants[k] = v
    #     constants_to_names[v] = k

    ops_dict["graph_top_trace"] = graph_top_trace
    ops_dict["constants"] = constants
    ops_dict["constants_to_names"] = constants_to_names

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


def stringify_type(op, if_flist=False):
    def loop(vals, typs, names):
        for i, t in enumerate(typs):
            if not isinstance(typs[i], str):
                typs[i] = str(typs[i].annotation_str)
        for i, v in enumerate(vals):
            if "Device" in typs[i] and v is not None:
                vals[i] = v.name
            if isinstance(v, tuple):
                vals[i] = list(v)
        for i, n in enumerate(names):
            if names[i] is not None:
                names[i] = names[i].replace("$", "%")

    def _stringify_type(op):
        if "caller_op_id" in op:
            del op["caller_op_id"]
        loop(op["args"], op["args types"], op.get("args names", []))
        loop(op["returns"], op["returns types"], op.get("returns names", []))
        if if_flist:
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
    # constants_to_names = ops_dict["constants_to_names"]
    # graph_args = ops_dict["graph"]["$args"]
    graph = ops_dict["graph"]

    alloc_ptr_to_alloc = {
        f"tensor_ptr_{a['addr']}": id_to_op[a["op_id"]]
        for a in ops_dict["allocations_list"]
        if a["fn_name"] == "allocate"
    }
    names_to_allocs = {}
    for ptr, ptr_name in ptr_addr_to_name.items():
        if alloc := alloc_ptr_to_alloc.get(ptr):
            names_to_allocs[ptr_name] = alloc

    call_chains = defaultdict(list)
    qualified_call_chains = {}
    for alloc_name, call in names_to_allocs.items():
        size = call["size"]
        while caller_op_id := call.get("caller_op_id"):
            call_chains[alloc_name].append(caller_op_id)
            call = id_to_op[caller_op_id]

        call_chain = list(reversed(call_chains[alloc_name]))
        qualified_call_chain = []
        op = id_to_op[call_chain[1]]
        if len(op["returns names"]) == 1 and op["returns names"][0] in graph:
            out_name = op["returns names"][0]
            full_call = graph[out_name]
            qualified_call_chain.append(f"{out_name}: {full_call.replace('$', '%')}")

        for op_id in call_chain[2:]:
            op = id_to_op[op_id]
            # if op["schema"] != "no schema":
            #     fn = f"{op['fn_name']}({{}})"
            # arg_names = []
            # for i, arg in enumerate(op["args"]):
            #     if not isinstance(arg, str):
            #         if (
            #             isinstance(arg, tuple)
            #             and isinstance(op["args types"][i], ListType)
            #             and isinstance(
            #                 op["args types"][i].getElementType(), TensorType
            #             )
            #         ):
            #             arg_names.append(op["args names"][i])
            #         else:
            #             assert arg in constants_to_names
            #             arg_names.append(constants_to_names[arg])
            #     elif "tensor_ptr" in arg:
            #         if arg in ptr_addr_to_name:
            #             ptr_name = ptr_addr_to_name[arg]
            #             if ptr_name in allocs:
            #                 arg_names.append(ptr_name)
            #             else:
            #                 assert ptr_name in constants or ptr_name in graph_args
            #                 arg_names.append(ptr_name)
            #         else:
            #             assert "empty_tensor" in arg
            #             arg_names.append(arg)
            #     else:
            #         arg_names.append(arg)
            # qualified_call_chain.append(fn.format(", ".join(arg_names)))

            if op["schema"] != "no schema":
                fn = f"{op['fn_name']}({{}})"
                args = []
                for arg in op["args"]:
                    if not isinstance(arg, str):
                        if isinstance(arg, tuple):
                            arg = list(arg)
                        args.append(str(arg))
                    elif "tensor_ptr" in arg:
                        if arg in ptr_addr_to_name:
                            name = ptr_addr_to_name[arg].replace("$", "%")
                            if name in names_to_allocs:
                                args.append(name)
                            elif alloc_name in constants:
                                args.append(name)
                        else:
                            assert "empty_tensor" in arg
                            args.append(arg)
                    else:
                        args.append(arg)
                qualified_call_chain.append(fn.format(", ".join(args)))
        qualified_call_chain.append(f"allocate({size})")
        qualified_call_chains[alloc_name] = qualified_call_chain

    assert len(qualified_call_chains) == len(names_to_allocs)
    assert len(qualified_call_chains) == len(
        [o for o in ops_dict["allocations_list"] if o["fn_name"] == "allocate"]
    )
    assert set(qualified_call_chains.keys()) == set(names_to_allocs.keys())

    grouped_allocs = defaultdict(list)
    unique_call_chains = defaultdict(list)
    for k, v in qualified_call_chains.items():
        grouped_allocs[v[0]].append(k)
        unique_call_chains[tuple(v)].append(k)

    def find_all_frees(op):
        if op["fn_name"] == "free":
            res = [op["addr"]]
        else:
            res = []
        for call in ls_or_empty(op, "calls"):
            res.extend(find_all_frees(call))
        return res

    # prepare allocs for export and do checking
    for chain, alloc_names in unique_call_chains.items():
        for alloc_name in alloc_names:
            alloc = names_to_allocs[alloc_name]
            root_caller_op_id = call_chains[alloc_name][-2]  # -1 is the model name
            root_caller = id_to_op[root_caller_op_id]
            alloc["root_caller_fn_name"] = root_caller["fn_name"]
            alloc["ptr_tag"] = int(alloc["addr"].split("_")[1])

            # assert that these allocs don't escape
            if len(alloc_names) > 1:
                assert "alloc" in alloc_name
                assert alloc_name not in ops_dict["outs"]
                all_frees = find_all_frees(root_caller)
                assert alloc["addr"] in all_frees

    sorted_call_chain = {}
    name_order = ops_dict["name_order"]
    for ptr_name, chain in sorted(
        qualified_call_chains.items(), key=lambda name_chain: name_order[name_chain[0]]
    ):
        sorted_call_chain[ptr_name] = chain

    return sorted_call_chain, dict(grouped_allocs), names_to_allocs


def print_call_chains(call_chains, grouped_allocs, names_to_allocs, file_name):
    call_chains = {k.replace("$", "%"): v for k, v in call_chains.items()}
    grouped_allocs = {k.replace("$", "%"): v for k, v in grouped_allocs.items()}
    names_to_allocs = {k.replace("$", "%"): v for k, v in names_to_allocs.items()}

    for name, output_alloc in names_to_allocs.items():
        names_to_allocs[name] = stringify_type(output_alloc, if_flist=False)

    with open(f"call_chains/{file_name}.json", "w") as f:
        json.dump(
            {
                "call_chains": call_chains,
                "grouped_allocs": grouped_allocs,
                "output_allocs": names_to_allocs,
            },
            f,
            indent=2,
        )

    for name, output_alloc in names_to_allocs.items():
        names_to_allocs[name] = stringify_type(output_alloc, if_flist=True)

    with open(f"call_chains/{file_name}.yml", "w") as f:
        yaml.dump(
            {
                "call_chains": call_chains,
                "grouped_allocs": grouped_allocs,
                "output_allocs": names_to_allocs,
            },
            f,
        )


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


def get_call_chains(ops_dict):
    ops_dict = label_pointers(ops_dict)
    sorted_call_chain, grouped_allocs, names_to_allocs = match_allocs_to_outs(ops_dict)
    return sorted_call_chain, grouped_allocs, names_to_allocs


def main(model_name):
    print(model_name)
    ops_dict = load_ops_yaml(model_name)
    ops_dict = parse_ops_dict(ops_dict)
    sorted_call_chain, grouped_allocs, names_to_allocs = get_call_chains(ops_dict)
    print_call_chains(sorted_call_chain, grouped_allocs, names_to_allocs, model_name)


def fix_yaml(model_name):
    f = open(f"profiles/{model_name}.yml", "r+")
    f_lines = f.readlines()
    gr = f_lines[0]
    if "graph:" in gr:
        return
    gr = gr.replace("graph(", "").replace("):\n", "")
    gr_fix = f"graph:\n  $args:\n    {gr}\n"
    f_lines[0] = gr_fix
    f.seek(0)
    f.writelines(f_lines)


def export_memory_reqs(
    ops_dict, named_allocations
) -> Dict[LiveRange, Tuple[RequiredAlloc, str]]:
    alloc_op_id_to_root_caller = {
        n["op_id"]: n["root_caller_fn_name"] for n in named_allocations.values()
    }
    mem_events_paired = defaultdict(list)
    mem_events = ops_dict["allocations_list"]
    for mem_ev in mem_events:
        ptr_addr, size, ts = mem_ev["addr"], mem_ev["size"], mem_ev["op_id"]
        mem_events_paired[ptr_addr].append((size, ts))

    lvr_to_req_plus_root_caller = {}
    for ptr_addr, mem_evts in mem_events_paired.items():
        if len(mem_evts) < 2:
            # warnings.warn(f"no free {mem_evts}")
            # hacky...
            alloc_size, alloc_ts = mem_evts[0]
            mem_evts.append((alloc_size, alloc_ts + 100))

        alloc_evt, free_evt = mem_evts
        alloc_size, alloc_ts = alloc_evt
        free_size, free_ts = free_evt
        assert alloc_size == free_size
        assert alloc_ts < free_ts
        req = RequiredAlloc(LiveRange(alloc_ts, free_ts), alloc_size, ptr_addr)
        lvr_to_req_plus_root_caller[req.lvr] = namedtuple(
            "lvr_to_req_plus_root_caller", ["req", "root_caller_fn_name"]
        )(req, alloc_op_id_to_root_caller[alloc_ts])
    return lvr_to_req_plus_root_caller


if __name__ == "__main__":
    main("alexnet.x1")
