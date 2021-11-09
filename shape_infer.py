from collections import defaultdict
from pprint import pprint
from typing import List

import torch
from torch import nn
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr, wl


# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger('foo').debug('bah')


def get_symbolic_dim_names(
        frozen_model, inp_shapes: List[List[int]], simplified, sym_shape_to_val_debug_name
):
    mod_inputs = list(frozen_model.graph.inputs())
    if len(mod_inputs) == len(inp_shapes):
        for i, mod_input in enumerate(mod_inputs):
            if inp_shapes[i] is not None:
                mod_input.setType(mod_input.type().with_sizes(inp_shapes[i]))

    all_sizes = []
    for n in frozen_model.graph.nodes():
        if n.kind() == "prim::Constant":
            continue
        try:
            sym_sizes = [
                "(" + (simplified[sym_shape_to_val_debug_name[s]] if s < 0 else str(s)) + ")"
                for s in n.output().type().symbolic_sizes()
                # if s < 0
            ] 
            # + [f'Symbol["{n.output().debugName()}"]']

            print(n.kind(), "*".join(sym_sizes))
            all_sizes.append(f"({'*'.join(sym_sizes)})")
            # print(n.kind(), "[" + ", ".join(sym_sizes) + "]")
        except:
            pass

    # print("+".join(all_sizes))

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
            elif inp.node().kind() == "aten::div":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} / {res[b_name]})"
            elif inp.node().kind() == "aten::Int":
                a = inp.node().input() 
                return f"{res[a.debugName()]}"
            elif inp.node().kind() == "prim::If":
                cond = dfs(inp.node().input())
                true_clause = dfs(list(list(inp.node().blocks())[0].outputs())[0])
                false_clause = dfs(list(list(inp.node().blocks())[1].outputs())[0])
                return f"If[{cond}, {true_clause}, {false_clause}]"
            elif inp.node().kind() == "aten::remainder":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"Mod[{res[a_name]}, {res[b_name]}]"
            elif inp.node().kind() == "aten::eq":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} == {res[b_name]})"
            else:
                print(inp.node())
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

def next_multiple(n, k):
    return n + (k - n % k)


from typing import List, Optional


def check_non_negative(array: List[int]) -> bool:
    # TODO: look into rewriting with early return and getting loop unrolling to fire
    non_negative = False
    for val in array:
        if val < 0:
            non_negative = True
    return non_negative

def check_shape_forward(input: List[int], weight_sizes: List[int], bias: Optional[List[int]], stride: List[int],
                        padding: List[int], dilation: List[int], groups: int):
    k = len(input)
    weight_dim = len(weight_sizes)

    # TODO: assertions could be expanded with the error messages
    assert not check_non_negative(padding), "negative padding"
    assert not check_non_negative(stride), "negative stride"

    assert weight_dim == k, "weight_dim != k"
    assert weight_sizes[0] >= groups, "weight sizes[0] not >= groups"
    assert (weight_sizes[0] % groups) == 0, "weight sizes not divisable by groups"
    # only handling not transposed
    assert input[1] == weight_sizes[1] * groups, "weight sizes[1] != groups"
    assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])

    for i in range(k, k):
        assert (input[i] + 2 * padding[i - 2]) >= (
                dilation[i - 2] * (weight_sizes[i] - 1) + 1), "kernel size can't be bigger than input size"


# graph(%self : __torch__.torchvision.models.inception.___torch_mangle_103.Inception3,
#       %x.1 : Tensor(SS(-25), SS(-26), SS(-27), SS(-28))):
#   %4158 : int[] = prim::Constant[value=[1, 0]]()
#   %4157 : int[] = prim::Constant[value=[0, 1]]()
#   %4156 : int[] = prim::Constant[value=[3, 0]]()
#   %4155 : int[] = prim::Constant[value=[0, 3]]()
#   %4154 : int[] = prim::Constant[value=[3, 3]]()
#   %4153 : int[] = prim::Constant[value=[1, 1]]()
#   %4152 : int[] = prim::Constant[value=[0, 0]]()
#   %4151 : int[] = prim::Constant[value=[2, 2]]()
#   %58 : int = prim::Constant[value=-1]()
#   %15 : int = prim::Constant[value=1]() # /home/mlevental/miniconda3/envs/pytorch_shape_inference_memory_planning/lib/python3.8/site-packages/torchvision-0.12.0a0+dbb7679-py3.8-linux-x86_64.egg/torchvision/models/inception.py:131:45
#   %self.aux_logits : bool = prim::Constant[value=1]()
#   %self.fc.bias : Float(1000, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
#   %self.fc.weight : Float(1000, 2048, strides=[2048, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
#   %self.Conv2d_1a_3x3.conv.bias : NoneType = prim::Constant()
#   %self.transform_input : bool = prim::Constant[value=0]()
#   %self.Conv2d_1a_3x3.conv.weight_fused_bn : Float(32, 3, 3, 3, strides=[27, 9, 3, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
#   %self.Conv2d_1a_3x3.conv.bias_fused_bn.1 : Float(32, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
#   %x.46 : Tensor(SS(-25), 32, SS(-29), SS(-30)) = aten::conv2d(%x.1, %self.Conv2d_1a_3x3.conv.weight_fused_bn, %self.Conv2d_1a_3x3.conv.bias_fused_bn.1, %4151, %4152, %4153, %15) # /home/mlevental/dev_projects/pytorch_19/torch/nn/modules/conv.py:443:15

# this is not handling transposed convolution yet
def conv_output_size(input_size: List[int], weight_size: List[int], bias: Optional[List[int]], stride: List[int],
                     padding: List[int], dilation: List[int], groups: int):
    check_shape_forward(input_size, weight_size, bias, stride, padding, dilation, groups)

    has_dilation = len(dilation) > 0
    dim = len(input_size)
    output_size: List[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])
    output_size.append(weight_size[weight_output_channels_dim])

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        # most common case of just enough padding to fit kernel and stride evenly
        # paves the input
        if (input_size[d] + (2 * padding[d - 2]) - (kernel - 1)) % stride[d - 2] == 0:
            append_value = int((input_size[d] + (2 * padding[d - 2]) - (kernel - 1)) / stride[d - 2])
        else:
            append_value = ((input_size[d] + (2 * padding[d - 2]) - (kernel - 1)) // stride[d - 2])
        output_size.append(append_value)

    return output_size

# print(conv_output_size([1,3, 10, 10], [3, 32, 3, 3], bias=None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1))

# exit()


def get_shape_compute_elias(model, inp_shapes: List[List[int]]):
    model_frozen = torch.jit.freeze(torch.jit.script(model.eval()))
    # print(model_frozen.graph)

    # until https://github.com/pytorch/pytorch/issues/65643 lands to clean up control flow..
    torch._C._jit_pass_propagate_shapes_on_graph(model_frozen.graph)
    torch._C._jit_pass_peephole(model_frozen.graph)
    torch._C._jit_pass_constant_propagation(model_frozen.graph)

    # unused tuple output, manipulate so we are just returning tensor
    model_name = model._get_name()
    if model_name in {"Inception3"}:
        output = next(model_frozen.graph.outputs())
        assert output.node().kind() == "prim::TupleConstruct"
        model_frozen.graph.eraseOutput(0)
        model_frozen.graph.registerOutput(next(output.node().inputs()))
        model_frozen.graph.findNode("aten::warn").destroy()
        torch._C._jit_pass_dce(model_frozen.graph)

    if model._get_name() == "deeplab":
        # remove error handling
        for node in model_frozen.graph.findAllNodes("prim::RaiseException"):
            node.destroy()
        torch._C._jit_pass_dce(model_frozen.graph)
        torch._C._jit_pass_remove_mutation(model_frozen.graph)
        torch._C._jit_pass_peephole(model_frozen.graph)
        # TODO - need to add handling of a couple ops, doen't quite work yet

    inps = list(model_frozen.graph.inputs())
    inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))
    shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(model_frozen.graph)
    g = shape_compute_graph.partial_eval_shape_graph()
    
    # here is the encoding of shape arithmetic from the input, represnted as TS graph
    for node in g.findAllNodes("prim::RaiseException"):
        node.destroy()
    torch._C._jit_pass_dce(g)

    remove_periods(g)
    return shape_compute_graph, model_frozen


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
    assumptions = []
    for v, rep in vars.items():
        if s != s.replace(v, rep):
            s = s.replace(v, rep)
            assumptions.append(f"Mod[{rep}, 32] == 0, {rep} > 0")
    fs = f"FullSimplify[{s}, Assumptions -> {{{', '.join(assumptions)}}}]"
    return WEval(wl.Expand(wlexpr(fs)))



class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

import torch
import torch.utils.cpp_extension

def register_op():
    op_source = """
    #include <torch/script.h>
    #include <ATen/ATen.h>
    #include <ATen/NativeFunctions.h>
    #include <ATen/native/UpSample.h>
    #include <c10/util/irange.h>

    using at::native::upsample::compute_output_size;
    using at::native::upsample::get_scale_value;

    torch::Tensor my_upsample_bilinear2d(
    const torch::Tensor& input,
    const torch::Tensor& output_size,
    bool align_corners,
    c10::optional<c10::ArrayRef<double>> scale_factors) {
        auto osize = compute_output_size(input.sizes(), output_size.sizes(), scale_factors);
        auto scale_h = get_scale_value(scale_factors, 0);
        auto scale_w = get_scale_value(scale_factors, 1);
        return at::upsample_bilinear2d(input, osize, align_corners, scale_h, scale_w);
    }


    TORCH_LIBRARY(my_ops, m) {
        m.def("my_upsample_bilinear2d", &my_upsample_bilinear2d);
    }
    """

    torch.utils.cpp_extension.load_inline(
        name="my_upsample_bilinear2d",
        cpp_sources=op_source,
        # extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
        is_python_module=False,
        verbose=True,
    )

    print(torch.ops.my_ops.my_upsample_bilinear2d)

# register_op()
import torchvision

if __name__ == "__main__":
    torch._C._jit_set_symbolic_shapes_test_mode(True)
    models = torchvision.models

    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'unet')
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #     in_channels=3, out_channels=1, init_features=32, pretrained=True)
    # model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN').netD
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
    print(model)
    model.eval()
    out = model(torch.randn(1,3, 128, 128))
    for k, v in out.items():
        print(v.shape)
    exit()
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11')
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # model = models.shufflenet_v2_x1_0()

    # model = models.inception_v3()
    # print(model)

    # print("resnet34")
    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet34")
    # print_shape_graph(model)

    shape_compute_graph, frozen_model = get_shape_compute_elias(model, [[None, None, None, None]])
    # print(frozen_model.graph)
    g = shape_compute_graph.partial_eval_shape_graph()
    print(g)
    output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
    # map from sym shape -> index
    sym_shape_to_index = {}
    sym_shape_to_val_debug_name = {}
    for index, output in enumerate(g.outputs()):
        sym_shape_to_index[output_sym_map[output]] = index
        sym_shape_to_val_debug_name[output_sym_map[output]] = output.debugName()
    
    print(sym_shape_to_val_debug_name)

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
            if isinstance(v, str):
                simplified[k] = map_str_to_mathematica(v, ss_map)
    print()
    print("simplified:")
    print()
    pprint(simplified)

    print()
    get_symbolic_dim_names(
        frozen_model, [[1, 3, 128, 128]], simplified, sym_shape_to_val_debug_name
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


