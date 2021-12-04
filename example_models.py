import inspect

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from transformers import BertTokenizer, BertModel, BertConfig

from modeling_gpt2 import GPT2LMHeadModel


def make_bert_input(batch_size, hw):
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
    tokens_tensor = torch.tensor([indexed_tokens * hw] * batch_size)
    segments_tensors = torch.tensor([segments_ids * hw] * batch_size)
    return tokens_tensor, segments_tensors


# - gpt2: 12
# - gpt2-medium: 24
# - gpt2-large: 36
# - gpt2-xl: 48


def make_gpt(name, batch_size, hw):
    with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokens_tensor, segments_tensors = make_bert_input(batch_size, hw)
        traced_model = torch.jit.trace(
            model, (tokens_tensor, segments_tensors), strict=False, check_trace=False
        )
        # torch.jit.save(traced_model, f"models/gpt2.pt")
        torch.jit.save(traced_model, f"models/gpt2.x{batch_size}.y{hw}.pt")

        # model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        # tokens_tensor, segments_tensors = make_bert_input(batch_size, hw)
        # traced_model = torch.jit.trace(
        #     model, (tokens_tensor, segments_tensors), strict=False, check_trace=False
        # )
        # # torch.jit.save(traced_model, f"models/gpt2-medium.pt")
        # torch.jit.save(traced_model, f"models/gpt2-medium.x{batch_size}.y{hw}.pt")
        #
        # model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        # tokens_tensor, segments_tensors = make_bert_input(batch_size, hw)
        # traced_model = torch.jit.trace(
        #     model, (tokens_tensor, segments_tensors), strict=False, check_trace=False
        # )
        # # torch.jit.save(traced_model, f"models/gpt2-large.pt")
        # torch.jit.save(traced_model, f"models/gpt2-large.x{batch_size}.y{hw}.pt")


def make_bert(name, batch_size, hw):
    with torch.no_grad():
        inp = make_bert_input(batch_size, hw)
        config = BertConfig(
            vocab_size_or_config_json_file=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            torchscript=True,
        )
        model = BertModel(config)
        model.eval()
        traced_model = torch.jit.trace(model, inp)
        # torch.jit.save(traced_model, f"models/large_bert.pt")
        torch.jit.save(traced_model, f"models/large_bert.x{batch_size}.y{hw}.pt")

        inp = make_bert_input(batch_size, hw)
        config = BertConfig(
            vocab_size_or_config_json_file=32000 // 2,
            hidden_size=768 // 2,
            num_hidden_layers=12 // 2,
            num_attention_heads=12 // 2,
            intermediate_size=3072 // 2,
            torchscript=True,
        )
        model = BertModel(config)
        model.eval()
        traced_model = torch.jit.trace(model, inp)
        # torch.jit.save(traced_model, f"models/medium_bert.pt")
        torch.jit.save(traced_model, f"models/medium_bert.x{batch_size}.y{hw}.pt")

        inp = make_bert_input(batch_size, hw)
        config = BertConfig(
            vocab_size_or_config_json_file=32000 // 4,
            hidden_size=768 // 4,
            num_hidden_layers=12 // 4,
            num_attention_heads=12 // 4,
            intermediate_size=3072 // 4,
            torchscript=True,
        )
        model = BertModel(config)
        model.eval()
        traced_model = torch.jit.trace(model, inp)
        # torch.jit.save(traced_model, f"models/small_bert.pt")
        torch.jit.save(traced_model, f"models/small_bert.x{batch_size}.y{hw}.pt")


def make_dcgan(name, batch_size, hw):
    model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN")
    gnet = model.getNetD()
    t = torch.rand((batch_size, 3, hw, hw))
    traced_model = torch.jit.trace(gnet, (t,))
    torch.jit.save(traced_model, f"models/dcgan.pt")
    # torch.jit.save(traced_model, f"models/dcgan.x{batch_size}.y{hw}.pt")


def make_unet(batch_size, hw):
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model.eval()
    x = torch.rand((batch_size, 3, hw, hw))
    y = torch.jit.trace(model, (x,))
    y.save(f"models/unet.x{batch_size}.y{hw}.pt")


def _vision_models():
    modelss = dict(inspect.getmembers(models, inspect.isfunction))
    detection_models = models.detection
    segmentation_models = models.segmentation
    modelss.update(dict(inspect.getmembers(detection_models, inspect.isfunction)))
    modelss.update(dict(inspect.getmembers(segmentation_models, inspect.isfunction)))
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


def make_toy_model(batch_size, hw):
    x = torch.rand((batch_size, 1, hw, hw))

    model = Net()
    model.eval()
    y = torch.jit.trace(model, (x,), strict=False)
    # y.save(f"models/toy_model.x{batch_size}.y{hw}.pt")
    y.save(f"models/toy_model.pt")


def make_vision_model(model_name, batch_size=1, hw=64):
    if "inception" in model_name:
        hw = 128
    if "alexnet" in model_name:
        hw = 64
    print(model_name)
    with torch.no_grad():
        model = vision_models(model_name).eval()
        x = torch.rand((batch_size, 3, hw, hw))
        y = torch.jit.trace(model, (x,), strict=False)
        y.save(f"models/{model_name}.pt")


import multiprocessing


def make_all_vision_models(batch_size, hw):
    from main import with_log

    modelss = _vision_models()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for name, model_fn in modelss.items():
            pool.apply_async(with_log, (make_vision_model, (name, batch_size, hw)))
        pool.close()
        pool.join()


def make_transformers():
    from main import with_log

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            for hw in [1, 2, 4, 8]:
                pool.apply_async(with_log, (make_bert, ("bert", batch_size, hw)))
                pool.apply_async(with_log, (make_gpt, ("gpt", batch_size, hw)))
        pool.close()
        pool.join()


def test_bert_jit_model():
    with torch.no_grad():
        inps = make_bert_input(5, 5)
        print(inps)
        model = torch.jit.load(
            "/home/mlevental/dev_projects/pytorch_memory_planning/models/small_bert.pt"
        )
        print(model.code)
        print(model.graph)
        # out = model(*inps)
    # print(out)


def test_alexnet():
    from main import with_log

    with_log(make_vision_model, ("alexnet", 1, 32))


if __name__ == "__main__":
    make_dcgan("dcgan", 1, 64)
    make_all_vision_models(1, 32)
    make_transformers()

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


# def make_transformer():
#     # download model from https://drive.google.com/file/d/1FKNQ4TYBXU5ArhLr3a5szjAymqNz4Gnm/view?usp=sharing
#     model = torch.jit.load(f"models/attention_is_all_you_need.x{batch_size}.y{hw}.pt")
#
#     froze = model
#     inputs = list(froze.graph.inputs())
#
#     # firs inp is batch size
#     inputs[1].setType(inputs[1].type().with_sizes([None, None]))
#     inputs[2].setType(inputs[1].type())
#
#     for node in froze.graph.findAllNodes("prim::Constant"):
#         ival = node.output().toIValue()
#         if not isinstance(ival, torch.Tensor):
#             continue
#         node.output().inferTypeFrom(ival)
#
#     torch._C._jit_pass_propagate_shapes_on_graph(froze.graph)
#     torch._C._jit_pass_canonicalize_for_shape_analysis(froze.graph)
#     compute = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
#         froze.graph
#     )
#     eval_g = compute.partial_eval_shape_graph()
#     mapping = {
#         "%" + n.debugName(): f"SS({v})"
#         for n, v in compute.graph_output_to_symbolic_shape_dim().items()
#     }
#
#     [node.destroy() for node in eval_g.findAllNodes("prim::RaiseException")]
#     torch._C._jit_pass_dce(eval_g)
#     if True:
#         # slightly cheating a bit here but this value will always be False
#         # or the model throws
#         # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Models.py#L45
#         # x.size(1) > pos_table lenght will throw
#         #
#         # TODO: rely on compiler
#         cons = eval_g.insertConstant(False)
#         ge = eval_g.findNode("aten::ge")
#         ge_inps = list(ge.inputs())
#         assert ge_inps[1].toIValue() == 200
#         assert ge_inps[0].node().kind() == "aten::__getitem__"
#         cons.node().moveBefore(ge)
#         ge.replaceAllUsesWith(cons.node())
#         torch._C._jit_pass_constant_propagation(eval_g)
#
#     eval_g.makeMultiOutputIntoTuple()
#     func = torch._C._create_function_from_graph("partial_eval_graph", eval_g)
#     # print(func.code)
#     # print(eval_g)
#     [node.destroy() for node in eval_g.findAllNodes("prim::RaiseException")]
#     torch._C._jit_pass_dce(eval_g)
#     # print(eval_g)
#     # print(mapping)
#     # TODO: assumption that the input(1) size < 200
#     # gets clean graph
#     return compute, eval_g, mapping, froze
