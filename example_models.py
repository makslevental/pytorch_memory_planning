import inspect

import torch
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet101
from transformers import BertTokenizer, BertModel, BertConfig


def make_resnets():
    models_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    for models_name in models_names:
        model = torch.hub.load("pytorch/vision:v0.10.0", models_name)
        model.eval()
        x = torch.rand((1, 3, 224, 244))
        y = torch.jit.trace(model, (x,))

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


def vision_models(name):
    modelss = dict(inspect.getmembers(models, inspect.isfunction))
    detection_models = models.detection
    segmentation_models = models.segmentation
    modelss.update(
        dict(inspect.getmembers(detection_models, inspect.isfunction))
    )
    modelss.update(
        dict(inspect.getmembers(segmentation_models, inspect.isfunction))
    )
    return modelss[name](pretrained=False)

if __name__ == "__main__":
    # make_transformer()
    vision_models("alexnet")
