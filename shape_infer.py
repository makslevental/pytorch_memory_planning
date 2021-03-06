from pprint import pprint
from typing import List

import torch
from torch import nn


# XXX: still in prototype
# class TestSymbolicShapeAnalysis(JitTestCase):
#     def setUp(self):
#         self.prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
#         torch._C._jit_set_symbolic_shapes_test_mode(True)
#
#     def tearDown(self):
#         torch._C._jit_set_symbolic_shapes_test_mode(self.prev_symbolic_shapes_test_enabled)
#
#     def test_shape_analysis(self):
#         @torch.jit.script
#         def foo(x, y):
#             return x * y
#
#         inputs = list(foo.graph.inputs())
#
#         def prop_shapes_on_graph(inp0, inp1):
#             inputs[0].setType(inputs[0].type().with_sizes(inp0))
#             inputs[1].setType(inputs[1].type().with_sizes(inp1))
#             torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
#
#         prop_shapes_on_graph([1, 6, 5], [1, 7, 1, 5])
#         FileCheck().check("1, 7, 6, 5").run(foo.graph)
#
#         # None implicitly creates a new symbolic symbol
#         prop_shapes_on_graph([None, None], [None, None, None])
#         output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()
#         inp0_shape = inputs[0].type().symbolic_sizes()
#         inp1_shape = inputs[1].type().symbolic_sizes()
#
#         # output shape dim 0 should be taken from the second inp dim0
#         # other two dims we cannot infer and are given a new symbolic shape
#         self.assertEqual(output_shape[0], inp1_shape[0])
#         self.assertFalse(output_shape[1] in inp0_shape + inp1_shape)
#         self.assertFalse(output_shape[2] in inp0_shape + inp1_shape)
#
#         # XXX: symbolic shapes are represented with an increasing counter of unique
#         # values, use `_new_symbolic_shape_symbol` api instead of specifying negative
#         # dimensions directly so there is no chance of collision between manual number
#         # and current counter value.
#         sym1 = torch._C._new_symbolic_shape_symbol()
#         sym2 = torch._C._new_symbolic_shape_symbol()
#         sym3 = torch._C._new_symbolic_shape_symbol()
#         prop_shapes_on_graph([sym1, 1, sym3], [1, sym2, sym3])
#         output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()
#         self.assertEqual(output_shape[0], sym1)
#         self.assertEqual(output_shape[1], sym2)
#         self.assertEqual(output_shape[2], sym3)
#
#     def test_shared_shape_graph(self):
#         @torch.jit.script
#         def foo(x, y):
#             return x * y, x / y
#
#         mul_node = foo.graph.findNode("aten::mul")
#         div_node = foo.graph.findNode("aten::div")
#
#         mul_graph = torch._C._jit_shape_compute_graph_for_node(mul_node)
#         div_graph = torch._C._jit_shape_compute_graph_for_node(div_node)
#         self.assertIsNotNone(mul_graph)
#         self.assertIs(mul_graph, div_graph)
#
#     def test_write(self):
#         @torch.jit.script
#         def foo(a, b):
#             return a * b
#
#         # broadcast appends cant be removed, so we bail on propagation
#         torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
#         FileCheck().check("Tensor = aten::mul").run(foo.graph)
#
#         @torch.jit.script
#         def foo(y):
#             x = [1, 2, 3, 4]
#             x[0] = 5
#             return y.view(x)
#
#         torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
#         FileCheck().check("Tensor = aten::view").run(foo.graph)
#
#     def test_if_propagation(self):
#         @torch.jit.script
#         def foo(i: int, z):
#             x = torch.ones([2, 3, 4, 5])
#             y = z.view([z.size(i), 3, 2, z.size(i)])
#             if i == 4:
#                 return x
#             else:
#                 return y
#
#         torch._C._jit_pass_constant_propagation(foo.graph)
#         torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
#         FileCheck().check("*, 3, 2, *").check("*, 3, *, *) = prim::If").run(foo.graph)
#
#     def test_unary_shape_functions(self):
#         def apply(fn):
#             return lambda x: fn(x)
#
#         unary_ops = [
#             torch.nn.functional.hardtanh,
#         ]
#         for fn in unary_ops:
#             t = torch.jit.trace(fn, (torch.rand([4, 4])))
#             ten_input = next(t.graph.inputs())
#             ten_input.setType(ten_input.type().with_sizes([2, 2]))
#             torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
#             self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])
#
#     def test_binary_shape_functions(self):
#         def apply(fn):
#             return lambda x, y: fn(x, y)
#
#         binary_ops = [
#             operator.__mul__,
#             operator.__truediv__,
#             operator.__gt__,
#             operator.__add__,
#         ]
#
#         for fn in binary_ops:
#             size_1 = [1, 4, 8]
#             size_2 = [4, 1, 8]
#             t = torch.jit.trace(fn, (torch.rand([4]), torch.rand([4])))
#             inputs = list(t.graph.inputs())
#             inputs[0].setType(inputs[0].type().with_sizes(size_1))
#             inputs[1].setType(inputs[1].type().with_sizes(size_2))
#             torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
#             self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])
#
#     def test_size_and_sizes(self):
#         @torch.jit.script
#         def foo(x, y):
#             return x.view(y.size(0), 8, y.size(-1))
#
#         @torch.jit.script
#         def foo2(x, y):
#             return x.view(y.size())
#
#         for graph in [foo.graph, foo2.graph]:
#             inputs = list(graph.inputs())
#             sym1 = torch._C._new_symbolic_shape_symbol()
#
#             inputs[1].setType(inputs[1].type().with_sizes([5, 8, sym1]))
#             torch._C._jit_pass_propagate_shapes_on_graph(graph)
#             self.assertEqual(next(graph.outputs()).type().symbolic_sizes(), [5, 8, sym1])
#
#     def test_adaptive_avg_pool2d(self):
#         inps = [
#             [(1, 64, 8, 9), (5, 7)],
#             [(1, 64, 10, 9), (7)],
#             [(1, 64, 10, 9), (5, None)],
#             [(1, 8, 4, 3), (None, None)],
#             [(1, 8, 4, 3), (None, 5)],
#         ]
#
#         for inp in inps:
#             t = torch.randn(*inp[0])
#             out_size = torch.nn.functional.adaptive_avg_pool2d(t, inp[1]).size()
#
#             def foo(x):
#                 return torch.nn.functional.adaptive_avg_pool2d(x, inp[1])
#
#             fn = torch.jit.trace(foo, (t,))
#             torch._C._jit_erase_non_input_shape_information(fn.graph)
#             torch._C._jit_pass_peephole(fn.graph)
#             torch._C._jit_pass_constant_propagation(fn.graph)
#             self.checkShapeAnalysis(out_size, fn.graph, assert_propagation=True)
#
#     def test_arange_shape(self):
#         # no opinfo for tensor constructors
#         inps = [
#             (10,),
#             (10, 10),
#             (0, 10),
#             (0, 1000),
#             (1, -1, -1),
#             (1, 0, -1),
#             (1, 2, 1),
#             (0.6, 0.89, 0.1),
#             (1, 10, 0.3),
#             (1, 10, 4),
#             (0.6, 0.7, 0.8),
#             (1, 10, 0.3),
#             # (True,),  TODO: https://github.com/pytorch/pytorch/issues/63405
#             # (False,), TODO: https://github.com/pytorch/pytorch/issues/63405
#             (0, 5),
#             (0, 5, 2),
#             (0, 5 + 1e-6),
#             (0, 5 - 1e-6),
#             (10, -1 + 1e-6, -1),
#             (10, -1, -1),
#             (10, -1 - 1e-6, -1),
#         ]
#
#         for inp in inps:
#             funcs_template = dedent('''
#             def func():
#                 return torch.arange({args})
#             ''')
#
#             inp_s = str(inp)[1:-1]  # remove tuple parens
#             funcs_str = funcs_template.format(args=inp_s)
#             scope = {}
#             execWrapper(funcs_str, globals(), scope)
#             cu = torch.jit.CompilationUnit(funcs_str)
#             self.checkShapeAnalysis(list(cu.func().size()), cu.func.graph, assert_propagation=True, constant_prop=False)
#
#     def test_shape_embedding_bag(self):
#         # TODO: merge into opinfos, having difficulties there
#         with torch.no_grad():
#             def make_arg(shape, low=None, high=None):
#                 return make_tensor(shape, device='cpu', dtype=torch.int64,
#                                    low=low, high=high, requires_grad=False)
#
#             nn_inps = (
#                 (make_arg((40,), 0, 9), torch.nn.Embedding(20, embedding_dim=64, max_norm=1.0)),
#                 (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 20, sparse=True)),
#                 (make_arg(()), torch.nn.Embedding(0, 0, sparse=True)),
#                 (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 0, sparse=True)),
#                 (make_arg((4,), 0, 21), torch.nn.Embedding(22, 5, max_norm=1.0)),
#                 (make_arg((2,), 0, 1), torch.nn.Embedding.from_pretrained(torch.arange(6.).view(2, 3), max_norm=2.,
#                                                                           norm_type=.5, scale_grad_by_freq=False,
#                                                                           sparse=True)),
#             )
#
#             for inp, module in nn_inps:
#                 kwargs = {
#                     "weight": module.weight.detach(),
#                     "padding_idx": module.padding_idx,
#                     "max_norm": module.max_norm,
#                     "norm_type": module.norm_type,
#                     "scale_grad_by_freq": module.scale_grad_by_freq,
#                     "sparse": module.sparse,
#                 }
#
#                 out_size = torch.nn.functional.embedding(inp, **kwargs).size()
#
#                 def foo(x):
#                     return torch.nn.functional.embedding(inp, **kwargs)
#
#                 fn = torch.jit.trace(foo, (inp.detach(),), check_trace=False)
#
#                 self.checkShapeAnalysis(out_size, fn.graph, assert_propagation=True, constant_prop=False)
#
#     def test_partial_eval_graph_conv(self):
#         mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
#         shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
#         output_sizes = mm.graph.findNode("aten::conv2d").output().type().symbolic_sizes()
#         # calculating 0, 2 and 3 index
#         for i in [0, 2, 3]:
#             self.assertTrue(output_sizes[i] < 0)
#         self.assertTrue(output_sizes[1] >= 0)
#         g = shape_compute_graph.partial_eval_shape_graph()
#         # to make into a jit function cant have multiple outputs
#         g.makeMultiOutputIntoTuple()
#         func = torch._C._create_function_from_graph("partial_eval_graph", g)
#         inp = torch.randn(20, 16, 5, 10)
#         output = func([20, 16, 5, 10])
#         output_eager = list(mm(inp).size())
#         for o, oe in zip(output, output_eager[0:1] + output_eager[2:]):
#             self.assertEqual(o, oe)
#
#     def test_partial_eval_stitching(self):
#         conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#         conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#
#         mod = torch.jit.freeze(torch.jit.script(nn.Sequential(conv1, max_pool, conv2).eval()))
#
#         conv1_output = conv1(torch.rand(1, 3, 224, 224))
#         max_pool_ouput = max_pool(conv1_output)
#         conv2_output = conv2(max_pool_ouput)
#
#         shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
#         g = shape_compute_graph.partial_eval_shape_graph()
#         self.assertTrue(len(list(g.inputs())) == 1)
#         output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
#         # map from sym shape -> index
#         sym_shape_to_index = {}
#         for index, output in enumerate(g.outputs()):
#             sym_shape_to_index[output_sym_map[output]] = index
#
#         g.makeMultiOutputIntoTuple()
#         func = torch._C._create_function_from_graph("partial_eval_graph", g)
#         sym_outputs = func([1, 3, 224, 224])
#         nodes = [mod.graph.findNode("aten::max_pool2d")] + list(mod.graph.findAllNodes("aten::conv2d"))
#         output_shapes = [max_pool_ouput, conv1_output, conv2_output]
#
#         for node, output_shape in zip(nodes, output_shapes):
#             output_type_sizes = node.output().type().symbolic_sizes()
#             for i, sym_shape in enumerate(output_type_sizes):
#                 if sym_shape >= 0:
#                     self.assertEqual(sym_shape, output_shape.size(i))
#                 else:
#                     sym_shape_index = sym_shape_to_index[sym_shape]
#                     self.assertEqual(sym_outputs[sym_shape_index], output_shape.size(i))
#
#     def test_refinement_through_graph_stitching(self):
#         class TwoConvs(torch.nn.Module):
#             def __init__(self):
#                 super(TwoConvs, self).__init__()
#                 self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#                 self.conv2 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#
#             def forward(self, x):
#                 a = self.conv1(x)
#                 b = self.conv2(x)
#                 return a + b
#
#         mod = torch.jit.freeze(torch.jit.script(TwoConvs()).eval())
#         inp_tensor = list(mod.graph.inputs())[1]
#         inp_tensor.setType(inp_tensor.type().with_sizes([None, None, None, None]))
#         torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
#         outs = list(next(mod.graph.outputs()).node().inputs())
#         out1 = outs[0].type().symbolic_sizes()
#         out2 = outs[1].type().symbolic_sizes()
#         self.assertTrue(out1[2] != out2[2])
#         self.assertTrue(out1[3] != out2[3])
#         # by joining partial eval graphs of both convs we are able to recognize the output shapes
#         # are equivalent
#         torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
#         out1 = outs[0].type().symbolic_sizes()
#         out2 = outs[1].type().symbolic_sizes()
#         self.assertEqual(out1, out2)


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


def get_symbolic_dim_names(mod, inp_shapes: List[List[int]] = []):
    print(type(mod))
    mod = torch.jit.freeze(torch.jit.script(mod.eval()))
    # print(mod.graph)

    mod_inputs = list(mod.graph.inputs())
    if len(mod_inputs) == len(inp_shapes):
        for i, mod_input in enumerate(mod_inputs):
            if inp_shapes[i] is not None:
                mod_input.setType(mod_input.type().with_sizes(inp_shapes[i]))

    torch._C._jit_pass_remove_mutation(mod.graph)
    torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
    # torch._C._jit_pass_peephole(mod.graph)
    # torch._C._jit_pass_constant_propagation(mod.graph)
    # print(mod.graph)

    for n in mod.graph.nodes():
        if n.kind() == "prim::Constant":
            continue
        try:
            print(
                "kind: ", n.kind(), "; sym sizes: ", n.output().type().symbolic_sizes()
            )
        except:
            pass


def basic_examples():
    conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    max_pool = torch.nn.MaxPool2d(
        kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
    )
    conv2 = nn.Conv2d(
        64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )

    # get_symbolic_dim_names(conv1)
    # get_symbolic_dim_names(max_pool, inp_shapes=[None, [-2, 64, -3, -4]])
    # get_symbolic_dim_names(conv2)

    # get_symbolic_dim_names(nn.Sequential(conv1, max_pool, conv2))

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
    get_symbolic_dim_names(model)

    # partial_eval_graph_conv()
    # test_partial_eval_stitching()
    # t = TestSymbolicShapeAnalysis()
    # t.setUp()
    # t.test_partial_eval_graph_conv()


def label_constants(graph):
    for n in graph.nodes():
        if n.kind() == "prim::Constant":
            n.output().setDebugName(n.output().debugName() + ".constant")


def backtrace_symbolic_outputs(op_shape_graph):
    print(op_shape_graph)

    def bfs(inp):
        if inp.node().kind() == "aten::__getitem__":
            print(inp.node().inputsAt(1).node().i("value"))
            return
        for next_inp in inp.node().inputs():
            print(next_inp)
            bfs(next_inp)

    bfs(op_shape_graph.return_node().input())
    print()


def print_shape_graph(model):
    frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
    torch._C._jit_pass_remove_mutation(frozen_model.graph)
    torch._C._jit_pass_propagate_shapes_on_graph(frozen_model.graph)
    torch._C._jit_pass_peephole(frozen_model.graph)
    torch._C._jit_pass_constant_propagation(frozen_model.graph)

    inps = list(frozen_model.graph.inputs())
    inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))

    label_constants(frozen_model.graph)

    shape_compute_graph = (
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
            frozen_model.graph
        )
    )
    g = shape_compute_graph.partial_eval_shape_graph()
    # print("Shape Compute Graph Before DCE\n\n")
    # print(g)
    for n in g.findAllNodes("prim::RaiseException"):
        n.destroy()
    torch._C._jit_pass_dce(g)
    # print("Shape Compute Graph \n\n")
    # print(g)
    op_shape_graphs = sorted(
        list(shape_compute_graph.partial_evaluated_graphs().items()),
        key=lambda x: x[0].output().debugName(),
    )
    for nn, gg in op_shape_graphs:
        for n in gg.findAllNodes("prim::RaiseException"):
            n.destroy()
        torch._C._jit_pass_dce(gg)
        label_constants(gg)
        # print(nn, gg)

    backtrace_symbolic_outputs(op_shape_graphs[0][1])
    print()
    # output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
    # print("output sym map\n\n")
    # print(output_sym_map)


def print_op_shape_graphs(model):
    frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
    for n in frozen_model.graph.nodes():
        if n.kind() == "prim::Constant":
            continue
        try:
            print(n.kind(), n.output().type().symbolic_sizes())
        except:
            pass


if __name__ == "__main__":
    torch._C._jit_set_symbolic_shapes_test_mode(True)

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
    # print_op_shape_graphs(model)

    # print("resnet34")
    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet34")
    print_shape_graph(model)
    #
    # print("resnet152")
    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152")
    # print_shape_graph(model)
