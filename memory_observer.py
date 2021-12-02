import glob
import multiprocessing
import os
import sys
import traceback
import warnings
from collections import defaultdict
from pprint import pprint

import ruamel.yaml

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
    import yaml

    with open(f"profiles/{model_name}.yml", "r") as stream:
        ops_dict = yaml.load(stream, Loader=yaml.CLoader)
    ops_dict["model_name"] = model_name
    return ops_dict


def with_log(fn, model_name):
    sys.stderr = open(f"logs/{model_name}.err", "a")
    sys.stdout = open(f"logs/{model_name}.log", "a")
    try:
        fn(model_name)
    except:
        traceback.print_exc()


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
        for o in (op.get("calls") or []):
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
            warnings.warn("no schema: " + op["fn_name"])

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
        compute_op_id for name, (_order, compute_op_id) in reversed(name_order.items()) if "alloc" not in name)
    graph_top["returns"] = id_to_op[last_compute_op_id]["returns"]
    graph_top["returns types"] = id_to_op[last_compute_op_id]["returns types"]
    graph_top["returns names"] = id_to_op[last_compute_op_id]["returns names"]

    for id, op in id_to_op.items():
        for o in op["calls"]:
            o["caller_op_id"] = op["op_id"]

    name_to_ptr = {}
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
                allocs = [(i, n) for i, n in enumerate(names) if "alloc" in n]
                others = [(i, n) for i, n in enumerate(names) if "alloc" not in n]

                for i, alloc_name in allocs:
                    for _, other_name in filter(lambda x: x[0] < i, others):
                        assert name_order[other_name] < name_order[alloc_name]
                    for _, other_name in filter(lambda x: x[0] > i, others):
                        assert name_order[alloc_name] < name_order[other_name]

                for i in range(len(others) - 1):
                    (_, return_name), (_, arg_name) = others[i], others[i + 1]
                    assert name_order[return_name] < name_order[arg_name]
                    # assert name_order[return_name] < name_order[alloc_name] < name_order[arg_name]
                    first_op = id_to_op[name_order[return_name][1]]
                    second_op = id_to_op[name_order[arg_name][1]]
                    assert ptr in first_op["returns"]
                    assert ptr in second_op["args"]
                names = [names[-1]]

        assert ptr not in ptr_addr_to_name
        ptr_addr_to_name[ptr] = names[0]

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
                elif isinstance(typ, DictType):
                    arg_or_ret_name = parent_args_or_rets_names[0]
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
                    if isinstance(arg_or_ret, str) and "tensor_ptr" in arg_or_ret:
                        raise "wtf"
                    warnings.warn(f'new constant {call["op_id"]} {arg_or_ret}')
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
                for arg in id_to_op[op_id]["args"]:
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
                qualified_call_chain.append(fn.format(", ".join(args)))
        qualified_call_chain.append(f"allocate({size})")
        call_chains[fn_name] = qualified_call_chain

    assert len(call_chains) == len(allocs)
    assert set(call_chains.keys()) == set(allocs.keys())
    unique_call_chains = defaultdict(list)
    for k, v in call_chains.items():
        unique_call_chains[tuple(v)].append(k)
    if len(unique_call_chains) != len(allocs):
        for c, ks in {
            c: ks for c, ks in unique_call_chains.items() if len(ks) > 1
        }.items():
            print(c, ks)
            for k in ks:
                pprint((k, allocs[k]["addr"], allocs[k]["size"]))

    sorted_call_chain = {}
    name_order = ops_dict["name_order"]
    for name, chain in sorted(call_chains.items(), key=lambda name_chain: name_order[name_chain[0]]):
        sorted_call_chain[name] = chain
    return sorted_call_chain, allocs


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


def main(model_name):
    print(model_name)
    ops_dict = load_ops_yaml(model_name)
    ops_dict = parse_ops_dict(ops_dict)
    ops_dict = label_pointers(ops_dict)
    call_chains, output_allocs = match_allocs_to_outs(ops_dict)
    print_call_chains(call_chains, output_allocs, model_name)


def fix_yaml(fp):
    f = open(fp, "r+")
    f_lines = f.readlines()
    print(f_lines[0])
    gr = f_lines[0]
    if "graph:" in gr:
        return
    gr = gr.replace("graph(", "").replace("):\n", "")
    gr_fix = f"graph:\n  $args:\n    {gr}\n"
    f_lines[0] = gr_fix
    f.seek(0)
    f.writelines(f_lines)


# def f(x):
#     return x*x
#
# if __name__ == '__main__':
#     with Pool(processes=4) as pool:         # start 4 worker processes
#         result = pool.apply_async(f, (10,)) # evaluate "f(10)" asynchronously in a single process
#     print(result.get())
# print(result.get(timeout=1))        # prints "100" unless your computer is *very* slow
#
# print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"
#
# it = pool.imap(f, range(10))
# print(next(it))                     # prints "0"
# print(next(it))                     # prints "1"
# print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow
#
# result = pool.apply_async(time.sleep, (10,))
# print(result.get(timeout=1))        # raises multiprocessing.TimeoutError

if __name__ == "__main__":
    # for i, fp in enumerate(glob.glob("profiles/*.yml")):

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for i, fp in enumerate(glob.glob("profiles/*.yml")):
            fix_yaml(fp)
            model_name = os.path.splitext(os.path.split(fp)[-1])[-2]
            pool.apply_async(with_log, (main, model_name))
        pool.close()
        pool.join()
    # for i, fp in enumerate(glob.glob("profiles/*.yml")):
    #     model_name = os.path.splitext(os.path.split(fp)[-1])[-2]
    #     main(model_name)

    # print_ops_dict(ops_dict, "resnet18")
    # find_out_variants(ops_dict, "resnet18")
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
