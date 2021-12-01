import inspect

import torch
# from torchvision import models
# from torchvision.models.segmentation import deeplabv3_resnet50
from torch import nn
from torch.nn import functional as F
from torch._C._autograd import DeviceType
# from torchvision import models
# from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
# from transformers import BertTokenizer, BertModel, BertConfig
# from torchvision import models
# from torchvision.models.segmentation import deeplabv3_resnet101
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input_shape = x.shape[-2:]
        # x = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=False)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def make_toy_model():
    x = torch.rand((1, 3, 32, 32))

    model = Net()
    model.eval()
    y = torch.jit.trace(model, (x,), strict=False)
    y.save(f"models/toy_model.pt")

def make_all_vision_models():
    modelss = _vision_models()
    for name, model_fn in modelss.items():
        print(name)
        with torch.no_grad():
            model = model_fn(pretrained=False).eval()
            print(model)
            x = torch.rand((1, 3, 52, 52))
            y = torch.jit.trace(model, (x,), strict=False)


if __name__ == "__main__":
    # pass
    # r = resnet18()
    # print(thnn_conv2d)
    # make_transformer()
    make_toy_model()
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
