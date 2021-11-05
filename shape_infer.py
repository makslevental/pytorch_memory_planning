from collections import defaultdict
from pprint import pprint
from typing import List

import torch
import torchvision
from torch import nn
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr


# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger('foo').debug('bah')


def test_partial_eval_stitching():
    conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    max_pool = torch.nn.MaxPool2d(
        kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
    )
    conv2 = nn.Conv2d(
        64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )

    mod = torch.jit.freeze(
        torch.jit.script(nn.Sequential(conv1, max_pool, conv2).eval())
    )

    # conv1_output = conv1(torch.rand(1, 3, 224, 224))
    # max_pool_ouput = max_pool(conv1_output)
    # conv2_output = conv2(max_pool_ouput)

    # shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
    shape_compute_graph = (
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
    )

    for n in mod.graph.nodes():
        try:
            print(n.output().type().symbolic_sizes())
            # print(n)
        except:
            pass

    g = shape_compute_graph.partial_eval_shape_graph()
    print(g)
    # self.assertTrue(len(list(g.inputs())) == 1)
    output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
    # map from sym shape -> index
    sym_shape_to_index = {}
    for index, output in enumerate(g.outputs()):
        sym_shape_to_index[output_sym_map[output]] = index

    pprint(output_sym_map)
    print(sym_shape_to_index)

    g.makeMultiOutputIntoTuple()
    func = torch._C._create_function_from_graph("partial_eval_graph", g)
    sym_outputs = func([1, 3, 224, 224])
    nodes = [mod.graph.findNode("aten::max_pool2d")] + list(
        mod.graph.findAllNodes("aten::conv2d")
    )
    # output_shapes = [max_pool_ouput, conv1_output, conv2_output]

    # for node, output_shape in zip(nodes, output_shapes):
    #     output_type_sizes = node.output().type().symbolic_sizes()
    #     for i, sym_shape in enumerate(output_type_sizes):
    #         if sym_shape >= 0:
    #             pass
    #             # self.assertEqual(sym_shape, output_shape.size(i))
    #         else:
    #             sym_shape_index = sym_shape_to_index[sym_shape]
    #             # self.assertEqual(sym_outputs[sym_shape_index], output_shape.size(i))


def partial_eval_graph_conv():
    inp = torch.randn(20, 16, 5, 10)
    mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
    shape_compute_graph = (
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
    )
    output_sizes = mm.graph.findNode("aten::conv2d").output().type().symbolic_sizes()
    print(output_sizes)
    # calculating 0, 2 and 3 index
    # for i in [0, 2, 3]:
    #     self.assertTrue(output_sizes[i] < 0)
    # self.assertTrue(output_sizes[1] >= 0)
    g = shape_compute_graph.partial_eval_shape_graph()
    # to make into a jit function cant have multiple outputs
    g.makeMultiOutputIntoTuple()
    func = torch._C._create_function_from_graph("partial_eval_graph", g)
    output = func([20, 16, 5, 10])
    output_eager = list(mm(inp).size())
    for o, oe in zip(output, output_eager[0:1] + output_eager[2:]):
        print(o, oe)


def get_symbolic_dim_names(
        frozen_model, inp_shapes: List[List[int]], simplified, sym_shape_to_val_debug_name
):
    mod_inputs = list(frozen_model.graph.inputs())
    if len(mod_inputs) == len(inp_shapes):
        for i, mod_input in enumerate(mod_inputs):
            if inp_shapes[i] is not None:
                mod_input.setType(mod_input.type().with_sizes(inp_shapes[i]))

    for n in frozen_model.graph.nodes():
        if n.kind() == "prim::Constant":
            continue
        try:
            sym_sizes = [
                simplified[sym_shape_to_val_debug_name[s]]
                for s in n.output().type().symbolic_sizes()
                if s < 0
            ]

            print("[" + ", ".join(sym_sizes) + "]")
            # print(n.kind(), "[" + ", ".join(sym_sizes) + "]")
        except:
            pass


def label_constants(graph):
    for n in graph.nodes():
        if n.kind() == "prim::Constant":
            n.output().setDebugName(n.output().debugName() + ".constant")


def remove_periods(graph):
    for inp in graph.inputs():
        try:
            inp.setDebugName(inp.debugName().replace(".", "_"))
        except:
            pass
    for inp in graph.outputs():
        try:
            inp.setDebugName(inp.debugName().replace(".", "_"))
        except:
            pass
    for n in graph.nodes():
        for inp in n.inputs():
            try:
                inp.setDebugName(inp.debugName().replace(".", "_"))
            except:
                pass
        for inp in n.outputs():
            try:
                inp.setDebugName(inp.debugName().replace(".", "_"))
            except:
                pass


def tree():
    return defaultdict(tree)


def dicts(t):
    if not isinstance(t, defaultdict):
        return t
    return {k: dicts(t[k]) for k in t}


def backtrace_symbolic_outputs(op_shape_graph):
    outputs = defaultdict(tree)

    def dfs(tre, inp):
        if inp.node().kind() == "prim::If":
            print(inp.node())

        val_name = "%" + inp.debugName()
        op = inp.node().kind()
        if inp.node().kind() == "prim::Constant":
            val = inp.node().i("value")
            tre[val_name, op] = val
        elif inp.node().kind() == "prim::Param":
            tre[val_name, op] = val_name
        else:
            op = str(inp.node()).split(" = ")[1].split("#")[0].strip()
            t = tre[val_name, op]

            for next_inp in inp.node().inputs():
                dfs(t, next_inp)

    for inp in op_shape_graph.return_node().inputs():
        dfs(outputs[f"%{inp.debugName()}"], inp)

    pprint(dicts(outputs))


def eval_backtrace_symbolic_outputs(op_shape_graph):
    def dfs(inp):
        val_name = inp.debugName()
        if inp.node().kind() == "prim::Constant":
            return inp.node().i("value")
        elif inp.node().kind() == "prim::Param":
            return val_name
        else:
            res = {}
            for next_inp in inp.node().inputs():
                res[next_inp.debugName()] = dfs(next_inp)

            if inp.node().kind() == "aten::__getitem__":
                obj, item = list(inp.node().inputs())
                obj_name, item_name = obj.debugName(), item.debugName()
                return f"{obj_name}[{res[item_name]}]"
            elif inp.node().kind() == "aten::sub":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} - {res[b_name]})"
            elif inp.node().kind() == "aten::add":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} + {res[b_name]})"
            elif inp.node().kind() == "aten::floordiv":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"Floor[{res[a_name]} / {res[b_name]}]"
            else:
                raise NotImplementedError

    res = {}
    for inp in op_shape_graph.return_node().inputs():
        res[inp.debugName()] = dfs(inp)

    return res


def get_shape_compute_graph(model, inp_shapes: List[List[int]]):
    frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
    if model._get_name() == "Inception3":
        output = next(frozen_model.graph.outputs())
        assert output.node().kind() == "prim::TupleConstruct"
        frozen_model.graph.eraseOutput(0)
        frozen_model.graph.registerOutput(next(output.node().inputs()))
        frozen_model.graph.findNode("aten::warn").destroy()
        torch._C._jit_pass_dce(frozen_model.graph)
    torch._C._jit_pass_remove_mutation(frozen_model.graph)
    torch._C._jit_pass_propagate_shapes_on_graph(frozen_model.graph)
    torch._C._jit_pass_peephole(frozen_model.graph)
    torch._C._jit_pass_constant_propagation(frozen_model.graph)

    mod_inputs = list(frozen_model.graph.inputs())
    if len(mod_inputs) == len(inp_shapes):
        for i, mod_input in enumerate(mod_inputs):
            if inp_shapes[i] is not None:
                mod_input.setType(mod_input.type().with_sizes(inp_shapes[i]))

    

    label_constants(frozen_model.graph)
    # remove_periods(frozen_model.graph)

    shape_compute_graph = (
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
            frozen_model.graph
        )
    )
    g = shape_compute_graph.partial_eval_shape_graph()
    # remove branches
    for n in g.findAllNodes("prim::RaiseException"):
        n.destroy()
    torch._C._jit_pass_dce(g)

    remove_periods(g)
    return shape_compute_graph, frozen_model


def get_op_shape_compute_graphs(shape_compute_graph):
    op_shape_graphs = sorted(
        list(shape_compute_graph.partial_evaluated_graphs().items()),
        key=lambda x: x[0].output().debugName(),
    )
    for _nn, gg in op_shape_graphs:
        for n in gg.findAllNodes("prim::RaiseException"):
            n.destroy()
        torch._C._jit_pass_dce(gg)

    return op_shape_graphs


def print_shape_graph(model):
    shape_compute_graph = get_shape_compute_graph(model)
    print(shape_compute_graph.partial_eval_shape_graph())


def print_op_shape_graphs(model):
    frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
    for n in frozen_model.graph.nodes():
        if n.kind() == "prim::Constant":
            continue
        try:
            print(n.kind(), n.output().type().symbolic_sizes())
        except:
            pass


def map_str_to_mathematica(s, vars={}):
    r = None
    for v, rep in vars.items():
        if s != s.replace(v, rep):
            r = rep
            s = s.replace(v, rep)

    new_vars = {"a": "k", "b": "l", "c": "m", "d": "n"}
    assumptions = f"{r} == 32{new_vars[r]}+1 && {new_vars[r]} > 0 && Element[{new_vars[r]}, Integers]]"
    print(assumptions)
    return WEval(wlexpr(f"FullSimplify[{s}, Assumptions -> {assumptions}"))


if __name__ == "__main__":
    torch._C._jit_set_symbolic_shapes_test_mode(True)
    models = torchvision.models

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

    # print("resnet34")
    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet34")
    # print_shape_graph(model)

    shape_compute_graph, frozen_model = get_shape_compute_graph(model, [[None, None, None, None]])
    g = shape_compute_graph.partial_eval_shape_graph()
    print(g)
    output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
    # map from sym shape -> index
    sym_shape_to_index = {}
    sym_shape_to_val_debug_name = {}
    for index, output in enumerate(g.outputs()):
        sym_shape_to_index[output_sym_map[output]] = index
        sym_shape_to_val_debug_name[output_sym_map[output]] = output.debugName()

    res = eval_backtrace_symbolic_outputs(g)
    ss_map = {
        "input_1[0]": "a",
        "input_1[1]": "b",
        "input_1[2]": "c",
        "input_1[3]": "d",
    }
    print()
    print("orig:")
    print()
    pprint(res)
    simplified = {}
    with WolframLanguageSession(
            kernel="/opt/Wolfram/WolframEngine/12.3/Executables/WolframKernel",
            # kernel_loglevel=logging.DEBUG,
            initfile="/home/mlevental/dev_projects/pytorch_memory_planning/initkernel.m",
    ) as session:
        WEval = session.evaluate
        for k, v in res.items():
            simplified[k] = map_str_to_mathematica(v, ss_map)
    print()
    print("simplified:")
    print()
    pprint(simplified)

    print()
    get_symbolic_dim_names(
        frozen_model, [[1, 3, 244, 244]], simplified, sym_shape_to_val_debug_name
    )

    # op_graphs = get_op_shape_compute_graphs(shape_compute_graph)
    # for nn, gg in op_graphs:
    #     print(nn, gg)
    #     # backtrace_symbolic_outputs(gg)
    #     print()

    #     backtrace_symbolic_outputs(gg)

    #
    # print("resnet152")
    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152")
    # print_shape_graph(model)
