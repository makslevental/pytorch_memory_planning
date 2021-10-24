import torch
from transformers import BertTokenizer, BertModel, BertConfig


def make_resnets():
    models_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    for models_name in models_names:
        model = torch.hub.load("pytorch/vision:v0.10.0", models_name)
        model.eval()
        x = torch.rand((1, 3, 224, 244))
        y = torch.jit.trace(model, (x,))

        y.save(f"models/{models_name}.pt")


def make_bert():
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
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768 // 4,
        num_hidden_layers=12 // 4,
        num_attention_heads=12 // 4,
        intermediate_size=3072 // 4,
        torchscript=True,
    )

    model = BertModel(config)
    model.eval()
    traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
    torch.jit.save(traced_model, "models/small_bert.pt")


def make_dcgan():
    model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN")
    gnet = model.getNetG()
    t = torch.rand((1, 64, 8, 8))
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
