from collections import namedtuple

import inspect

import torch
# from torchvision import models
# from torchvision.models.segmentation import deeplabv3_resnet50
import yaml
from torch import OptionalType, TensorType
from torch._C._autograd import DeviceType

import resnet


# from torchvision import models
# from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
# from transformers import BertTokenizer, BertModel, BertConfig
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet101
from transformers import BertTokenizer, BertModel, BertConfig


def make_resnets():
    models_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    for models_name in models_names:
        model = getattr(resnet, models_name)(pretrained=False)
        model.eval()
        x = torch.rand((1, 3, 224, 244))
        y = torch.jit.trace(model, (x,), _force_outplace=True)
        y._c = torch._C._freeze_module(y._c)
        # print(y.inlined_graph)
        y.save(f"models/{models_name}.pt")


def make_mobilenets():
    x = torch.rand((1, 3, 224, 244))

    model = models.mobilenet_v2()
    model.eval()
    y = torch.jit.trace(model, (x,))
    y.save(f"models/mobilenet_v2.pt")

    model = models.mobilenet_v3_small()
    model.eval()
    y = torch.jit.trace(model, (x,))
    y.save(f"models/mobilenet_v3_small.pt")

    model = models.mobilenet_v3_large()
    model.eval()
    y = torch.jit.trace(model, (x,))
    y.save(f"models/mobilenet_v3_large.pt")


def make_deeplab():
    x = torch.rand((1, 3, 224, 244))

    model = deeplabv3_resnet101()
    model.eval()
    y = torch.jit.trace(model, (x,), strict=False)
    y.save(f"models/deeplabv3_resnet101.pt")


def make_bert_input():
    enc = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensors


def make_bert():
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        torchscript=True,
    )

    inp = make_bert_input()

    model = BertModel(config)
    model.eval()
    traced_model = torch.jit.trace(model, inp)
    torch.jit.save(traced_model, "models/large_bert.pt")


def make_transformer():
    # download model from https://drive.google.com/file/d/1FKNQ4TYBXU5ArhLr3a5szjAymqNz4Gnm/view?usp=sharing
    model = torch.jit.load("models/attention_is_all_you_need.pt")

    froze = model
    inputs = list(froze.graph.inputs())

    # firs inp is batch size
    inputs[1].setType(inputs[1].type().with_sizes([None, None]))
    inputs[2].setType(inputs[1].type())

    for node in froze.graph.findAllNodes("prim::Constant"):
        ival = node.output().toIValue()
        if not isinstance(ival, torch.Tensor):
            continue
        node.output().inferTypeFrom(ival)

    torch._C._jit_pass_propagate_shapes_on_graph(froze.graph)
    torch._C._jit_pass_canonicalize_for_shape_analysis(froze.graph)
    compute = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
        froze.graph
    )
    eval_g = compute.partial_eval_shape_graph()
    mapping = {
        "%" + n.debugName(): f"SS({v})"
        for n, v in compute.graph_output_to_symbolic_shape_dim().items()
    }

    [node.destroy() for node in eval_g.findAllNodes("prim::RaiseException")]
    torch._C._jit_pass_dce(eval_g)
    if True:
        # slightly cheating a bit here but this value will always be False
        # or the model throws
        # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Models.py#L45
        # x.size(1) > pos_table lenght will throw
        #
        # TODO: rely on compiler
        cons = eval_g.insertConstant(False)
        ge = eval_g.findNode("aten::ge")
        ge_inps = list(ge.inputs())
        assert ge_inps[1].toIValue() == 200
        assert ge_inps[0].node().kind() == "aten::__getitem__"
        cons.node().moveBefore(ge)
        ge.replaceAllUsesWith(cons.node())
        torch._C._jit_pass_constant_propagation(eval_g)

    eval_g.makeMultiOutputIntoTuple()
    func = torch._C._create_function_from_graph("partial_eval_graph", eval_g)
    # print(func.code)
    # print(eval_g)
    [node.destroy() for node in eval_g.findAllNodes("prim::RaiseException")]
    torch._C._jit_pass_dce(eval_g)
    # print(eval_g)
    # print(mapping)
    # TODO: assumption that the input(1) size < 200
    # gets clean graph
    return compute, eval_g, mapping, froze


def make_dcgan():
    model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN")
    gnet = model.getNetD()
    t = torch.rand((1, 3, 128, 128))
    traced_model = torch.jit.trace(gnet, [t])
    torch.jit.save(traced_model, "models/dcgan.pt")


def make_unet():
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model.eval()
    x = torch.rand((1, 3, 256, 256))
    y = torch.jit.trace(model, (x,))
    y.save(f"models/unet.pt")


def _vision_models():
    modelss = dict(inspect.getmembers(models, inspect.isfunction))
    detection_models = models.detection
    segmentation_models = models.segmentation
    modelss.update(
        dict(inspect.getmembers(detection_models, inspect.isfunction))
    )
    modelss.update(
        dict(inspect.getmembers(segmentation_models, inspect.isfunction))
    )
    return modelss


def vision_models(name):
    return _vision_models()[name](pretrained=False)


def make_all_vision_models():
    modelss = _vision_models()
    for name, model_fn in modelss.items():
        print(name)
        with torch.no_grad():
            model = model_fn(pretrained=False).eval()
            print(model)
            x = torch.rand((1, 3, 52, 52))
            y = torch.jit.trace(model, (x,), strict=False)



def is_tensor_typ(typ):
    is_tensor = isinstance(typ, TensorType)
    is_opt = isinstance(typ, OptionalType)
    is_opt_tensor = is_opt and isinstance(typ.getElementType(), TensorType)
    return is_tensor or (is_opt and is_opt_tensor)


def parse_ops_yaml(fp, model_name):
    with open(fp, "r") as stream:
        yml = yaml.safe_load(stream)

    # json.dump(yml, open("ops.json", "w"), indent=2)

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
            if is_tensor_typ(typs[i]) and val == 0:
                vals[i] = "nullptr"

    def fill_in_type(op):
        id_to_op[op["op_id"]] = op

        if op["fn_name"] in {"allocate", "free"}: return
        schema = namedtuple('schema', ['arguments', 'returns'])(None, None)
        if op["schema"] != "no schema":
            schema = torch._C.parse_schema(op["schema"])
        else:
            # TODO: top-level -> ? maybe do something with return type of graph?
            for i, r in enumerate(op["args types"]):
                if r == "Tensor":
                    op["args types"][i] = TensorType.create_from_tensor(torch.tensor(-1))
            for i, r in enumerate(op["returns types"]):
                if r == "Tensor":
                    op["returns types"][i] = TensorType.create_from_tensor(torch.tensor(-1))

        loop(op["args"], op["args types"], schema.arguments)
        loop(op["returns"], op["returns types"], schema.returns)

        for o in (op.get("calls") or []):
            fill_in_type(o)

    for op in yml["trace"]:
        fill_in_type(op)

    return yml, id_to_op


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
    ptr_name_ptr_addr = ops_dict['ptr_name_ptr_addr']
    ptr_name_ptr_addr["nullptr"] = "nullptr"
    ptr_addr_to_name = {v: k for k, v in ptr_name_ptr_addr.items()}

    graph_top_trace = next(t for t in ops_dict["trace"] if t["fn_name"] == model_name)
    graph_args = list(map(list, ops_dict["graph"]["$args"].items()))

    # handle graph level
    for i, (a, t) in enumerate(graph_args):
        if t == "Tensor":
            graph_args[i][1] = TensorType.create_from_tensor(torch.tensor(-1))
    for i, (arg, typ) in enumerate(zip(graph_top_trace["args"], graph_top_trace["args types"])):
        if isinstance(typ, TensorType):
            graph_arg, graph_typ = graph_args[i]
            assert isinstance(graph_typ, TensorType)
            ptr_addr_to_name[arg] = graph_arg
            ptr_name_ptr_addr[graph_arg] = arg

    # handle top level for each ops
    for call in graph_top_trace["calls"]:
        if call["fn_name"] in {"allocate", "free"}: continue
        for i, (arg, arg_name) in enumerate(zip(call["args"], call["args names"])):

            typ = call["args types"][i]
            if isinstance(typ, TensorType):
                # ptrs aren't unique!!
                if arg in ptr_addr_to_name:
                    print("duplicate ptr", arg)
                ptr_addr_to_name[arg] = arg_name
                ptr_name_ptr_addr[arg_name] = arg
            if arg is not None and isinstance(typ, OptionalType) and isinstance(typ.getElementType(), TensorType):
                ptr_addr_to_name[arg] = arg_name
                ptr_name_ptr_addr[arg_name] = arg

    free_without_percolating = []

    def loop(vals, types, op_id, arg_or_return):
        for i, (arg, typ) in enumerate(zip(vals, types)):
            not_list = not isinstance(arg, list)
            in_dict = not_list and arg in ptr_addr_to_name
            is_tensor = isinstance(typ, TensorType)
            is_opt = isinstance(typ, OptionalType)
            is_opt_tensor = is_opt and isinstance(typ.getElementType(), TensorType)
            if not_list and in_dict:
                # double check on types
                assert is_tensor or (is_opt and is_opt_tensor)

            if is_tensor or (is_opt and is_opt_tensor):
                # either needs to be labeled or freed immediately (therefore no name)
                if arg is None:
                    assert is_opt and is_opt_tensor
                    continue
                if not in_dict:
                    # freed immediately; double check
                    free_without_percolating.append((i, arg, arg_or_return, op_id))

            if not_list and in_dict and is_tensor or (is_opt and is_opt_tensor):
                vals[i] = ptr_addr_to_name[arg]

    def label(op):
        if op["fn_name"] in {"allocate", "free"}: return
        loop(op["args"], op["args types"], op["op_id"], "args")
        loop(op["returns"], op["returns types"], op["op_id"], "returns")

        for o in (op.get("calls") or []):
            label(o)

    for op in ops_dict["trace"]:
        label(op)

    known_freed = set(op["addr"] for op in ops_dict["trace"] if op["fn_name"] == "free")

    def found_next_free(parent_id):
        return parent_id > 0 and any(c["fn_name"] == "free" for c in (id_to_op[parent_id].get("calls") or []))

    for arg_i, arg, arg_or_return, op_id in reversed(free_without_percolating):
        # TODO: think about this again
        if arg in known_freed: continue

        parent_id = op_id - 1
        while not found_next_free(parent_id):
            parent_id -= 1
        assert parent_id > - 1

        parent = id_to_op[parent_id]
        i, free_call = next((i, c) for i, c in enumerate(parent["calls"]) if c["fn_name"] == "free")
        assert free_call["op_id"] > op_id
        assert arg not in parent["returns"]
        known_freed.add(arg)

    return ops_dict


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
    labeled_ops_dict = label_pointers(ops_dict, id_to_op, "resnet18")
    find_out_variants(ops_dict, "resnet18")
    # parse_native_functions_yaml("/Users/mlevental/dev_projects/pytorch_shape_inference/aten/src/ATen/native/native_functions.yaml")
    # r = resnet18()
    # print(thnn_conv2d)
    # make_transformer()
    # make_resnets()

#
#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
# # (Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput)
#         if self.padding_mode != 'zeros':
#             return torch._C._nn._slow_conv2d_forward(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, self.kernel_size, bias, self.stride,
#                             _pair(0))[0]
#             # return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#             #                 weight, bias, self.stride,
#             #                 _pair(0), self.dilation, self.groups)
#         # return F.conv2d(input, weight, bias, self.stride,
#         #                 self.padding, self.dilation, self.groups)
#         return torch._C._nn._slow_conv2d_forward(input, weight, self.kernel_size, bias, self.stride,
#                         self.padding)[0]
#
