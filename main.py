import json
import os
from pprint import pprint

import torch
from transformers import BertTokenizer

from example_models import make_mobilenets, make_bert, make_deeplab, make_dcgan
from plotting import make_memory_map, plot_results
from profile_models import (
    get_required_mem_allocs,
    profile_model,
    get_mem_events, load_model, analyze_model,
)
from strategies import (
    greedy_by_size,
    bump_allocator,
    solve_cp,
    solve_mip, save_planned_allocs, calculate_high_watermark, gergov, verify_allocation, mincost_flow,
    greedy_by_longest,
)


def plan_greedy_strats(req_mem_allocs, name):
    planned_allocs = greedy_by_size(req_mem_allocs)
    # print("greedy_by_size_with_smallest_gap valid", verify_allocation(planned_allocs))
    print(f"{name}", "Greedy by Size", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.greedy_by_size_with_smallest_gap")

    # planned_allocs = greedy_by_size(
    #     req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    # )
    # print(f"{name}", "greedy_by_size_with_first_gap", calculate_high_watermark(planned_allocs), sep=",")
    # # make_memory_map(planned_allocs, f"{name}.greedy_by_size_with_first_gap")
    #
    planned_allocs = greedy_by_longest(req_mem_allocs)
    print(f"{name}", "Greedy by Longest", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.greedy_by_longest_with_smallest_gap")
    #
    # planned_allocs = greedy_by_longest(
    #     req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    # )
    # print(f"{name}", "greedy_by_longest_with_first_gap", calculate_high_watermark(planned_allocs), sep=",")
    # # make_memory_map(planned_allocs, f"{name}.greedy_by_longest_with_first_gap")


def plan_integer_programming_starts(req_mem_allocs, name):
    planned_allocs = solve_cp(req_mem_allocs)
    # print("solve_cp valid", verify_allocation(planned_allocs))
    print(f"{name}", "Symbolic CSP", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.solve_cp")

    planned_allocs = solve_mip(req_mem_allocs)
    # print("solve_mip valid", verify_allocation(planned_allocs))
    print(f"{name}", "Symbolic MIP", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.solve_mip")

def plan_other(req_mem_allocs, mem_events, name):
    planned_allocs = bump_allocator(mem_events)
    # print("bump valid", verify_allocation(planned_allocs))
    print(f"{name}", "Bump Allocator", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.bump")

    planned_allocs = mincost_flow(req_mem_allocs)
    # print("mincost_flow valid", verify_allocation(planned_allocs))
    print(f"{name}", "Min-cost Flow", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.mincost_flow")

    planned_allocs = gergov(req_mem_allocs)
    # print("gergov valid", verify_allocation(planned_allocs))
    print(f"{name}", "Gergov", calculate_high_watermark(planned_allocs), sep=",")
    make_memory_map(planned_allocs, f"{name}.gergov")


def make_maps(model, x, name):
    # trace_json = json.load(open(os.getcwd() + f"/traces/{name}.json"))
    # req_mem_allocs = get_required_mem_allocs(trace_json)
    # mem_events = get_mem_events(trace_json)

    req_mem_allocs, mem_events = analyze_model(model, x)
    # print("len req_mem_allocs", len(req_mem_allocs))

    plan_greedy_strats(req_mem_allocs, name)
    plan_integer_programming_starts(req_mem_allocs, name)
    plan_other(req_mem_allocs, mem_events, name)


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

if __name__ == "__main__":
    # for dir in ["models", "memory_maps", "traces"]:
    #     if not os.path.isdir(dir):
    #         os.mkdir(dir)
    # sh_obj = ctypes.cdll.LoadLibrary("runtime_patch/libruntime_patch.so")
    # make_bert()
    # make_dcgan()
    # make_resnets()
    # make_mobilenets()
    # make_deeplab()
    #
    for name in [
        # "resnet152",
        # "dcgan",
        # "mobilenet_v3_small",
        # "mobilenet_v3_large",
        # "mobilenet_v2",
        # "deeplabv3_resnet101",
        "attention_is_all_you_need"
    ]:
        model = load_model(os.getcwd() + f"/models/{name}.pt")
        if name == "small_bert" or name == "attention_is_all_you_need":
            inp = make_bert_input()
        else:
            inp = torch.rand((1, 3, 128, 128))
        make_maps(model, inp, f"{name},1x3x128x128")
    plot_results("res.csv")
    # make_maps(model, torch.rand((2, 3, 100, 100)), "resnet18.2x3x100x100")
    # make_maps(model, torch.rand((4, 3, 100, 100)), "resnet18.4x3x100x100")
    # make_maps(model, torch.rand((8, 3, 100, 100)), "resnet18.8x3x100x100")
    # make_maps(model, torch.rand((16, 3, 100, 100)), "resnet18.16x3x100x100")
    # solve_z3()
    # hook_alloc_cpu()
