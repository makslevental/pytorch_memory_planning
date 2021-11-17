import json
import os
import sys
from functools import partial
from pprint import pprint

import torch
from transformers import BertTokenizer

from example_models import make_mobilenets, make_bert, make_deeplab, make_dcgan, make_bert_input
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
    greedy_by_longest, find_gap, GapPriority,
)


def plan_greedy_strats(req_mem_allocs, name):
    planned_allocs = greedy_by_size(req_mem_allocs)
    # print("greedy_by_size_with_smallest_gap valid", verify_allocation(planned_allocs))
    print(f"{name}", "Greedy by Size", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.greedy_by_size_with_smallest_gap")

    planned_allocs = greedy_by_size(
        req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    )
    print(f"{name}", "greedy_by_size_with_first_gap", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.greedy_by_size_with_first_gap")

    planned_allocs = greedy_by_longest(req_mem_allocs)
    print(f"{name}", "Greedy by Longest", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.greedy_by_longest_with_smallest_gap")

    planned_allocs = greedy_by_longest(
        req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    )
    print(f"{name}", "greedy_by_longest_with_first_gap", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.greedy_by_longest_with_first_gap")


def plan_integer_programming_starts(req_mem_allocs, name):
    planned_allocs = solve_cp(req_mem_allocs)
    # print("solve_cp valid", verify_allocation(planned_allocs))
    print(f"{name}", "Symbolic CSP", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.solve_cp")

    # planned_allocs = solve_mip(req_mem_allocs)
    # # print("solve_mip valid", verify_allocation(planned_allocs))
    # print(f"{name}", "Symbolic MIP", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.solve_mip")


def plan_other(req_mem_allocs, mem_events, name):
    planned_allocs = bump_allocator(mem_events)
    print("bump valid", verify_allocation(planned_allocs))
    print(f"{name}", "Bump Allocator", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.bump")

    planned_allocs = mincost_flow(req_mem_allocs)
    # print("mincost_flow valid", verify_allocation(planned_allocs))
    print(f"{name}", "Min-cost Flow", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.mincost_flow")

    planned_allocs = gergov(req_mem_allocs)
    print("gergov valid", verify_allocation(planned_allocs))
    print(f"{name}", "Gergov", calculate_high_watermark(planned_allocs), sep=",")
    # make_memory_map(planned_allocs, f"{name}.gergov")


def make_maps(model, x, name):
    # trace_json = json.load(open(os.getcwd() + f"/traces/{name}.json"))
    # req_mem_allocs = get_required_mem_allocs(trace_json)
    # mem_events = get_mem_events(trace_json)

    req_mem_allocs, mem_events = analyze_model(model, x)
    # print("len req_mem_allocs", len(req_mem_allocs))

    plan_greedy_strats(req_mem_allocs, name)
    plan_integer_programming_starts(req_mem_allocs, name)
    # plan_other(req_mem_allocs, mem_events, name)


from contextlib import redirect_stdout
import torchvision.models as models


def vision_models():
    return {
        "resnet18": models.resnet18(pretrained=False),
        "alexnet": models.alexnet(pretrained=False),
        "squeezenet": models.squeezenet1_0(pretrained=False),
        "vgg16": models.vgg16(pretrained=False),
        "densenet": models.densenet161(pretrained=False),
        "inception": models.inception_v3(pretrained=False),
        "googlenet": models.googlenet(pretrained=False),
        "shufflenet": models.shufflenet_v2_x1_0(pretrained=False),
        "mobilenet_v2": models.mobilenet_v2(pretrained=False),
        "mobilenet_v3_large": models.mobilenet_v3_large(pretrained=False),
        "mobilenet_v3_small": models.mobilenet_v3_small(pretrained=False),
        "resnext50_32x4d": models.resnext50_32x4d(pretrained=False),
        "wide_resnet50_2": models.wide_resnet50_2(pretrained=False),
        "mnasnet": models.mnasnet1_0(pretrained=False),
        "efficientnet_b0": models.efficientnet_b0(pretrained=False),
        "efficientnet_b1": models.efficientnet_b1(pretrained=False),
        "efficientnet_b2": models.efficientnet_b2(pretrained=False),
        "efficientnet_b3": models.efficientnet_b3(pretrained=False),
        "efficientnet_b4": models.efficientnet_b4(pretrained=False),
        "efficientnet_b5": models.efficientnet_b5(pretrained=False),
        "efficientnet_b6": models.efficientnet_b6(pretrained=False),
        "efficientnet_b7": models.efficientnet_b7(pretrained=False),
        "regnet_y_400mf": models.regnet_y_400mf(pretrained=False),
        "regnet_y_800mf": models.regnet_y_800mf(pretrained=False),
        "regnet_y_1_6gf": models.regnet_y_1_6gf(pretrained=False),
        "regnet_y_3_2gf": models.regnet_y_3_2gf(pretrained=False),
        "regnet_y_8gf": models.regnet_y_8gf(pretrained=False),
        "regnet_y_16gf": models.regnet_y_16gf(pretrained=False),
        "regnet_y_32gf": models.regnet_y_32gf(pretrained=False),
        "regnet_x_400mf": models.regnet_x_400mf(pretrained=False),
        "regnet_x_800mf": models.regnet_x_800mf(pretrained=False),
        "regnet_x_1_6gf": models.regnet_x_1_6gf(pretrained=False),
        "regnet_x_3_2gf": models.regnet_x_3_2gf(pretrained=False),
        "regnet_x_8gf": models.regnet_x_8gf(pretrained=False),
        "regnet_x_16gf": models.regnet_x_16gf(pretrained=False),
        "regnet_x_32gf": models.regnet_x_32gf(pretrained=False),
    }


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
    # modelss = vision_models()
    # with open('res.csv', 'w') as f:
    #     with redirect_stdout(f):
    #         print("model,dims,strategy,size")
    #         # for name in [
    #         #     "resnet152",
    #         #     # "dcgan",
    #         #     # "mobilenet_v3_small",
    #         #     # "mobilenet_v3_large",
    #         #     # "mobilenet_v2",
    #         #     # "deeplabv3_resnet101",
    #         #     # "attention_is_all_you_need"
    #         #     "large_bert"
    #         # ]:
    #         #     model = load_model(os.getcwd() + f"/models/{name}.pt")
    #         for name, model in modelss.items():
    #             if name in {"small_bert", "large_bert", "attention_is_all_you_need"}:
    #                 inp = make_bert_input()
    #             else:
    #                 inp = torch.rand((1, 3, 128, 128))
    #             try:
    #                 make_maps(model, inp, f"{name},1x3x128x128")
    #             except Exception as e:
    #                 print(e, file=sys.stderr)
    plot_results("res.csv")
    # make_maps(model, torch.rand((2, 3, 100, 100)), "resnet18.2x3x100x100")
    # make_maps(model, torch.rand((4, 3, 100, 100)), "resnet18.4x3x100x100")
    # make_maps(model, torch.rand((8, 3, 100, 100)), "resnet18.8x3x100x100")
    # make_maps(model, torch.rand((16, 3, 100, 100)), "resnet18.16x3x100x100")
    # solve_z3()
    # hook_alloc_cpu()
