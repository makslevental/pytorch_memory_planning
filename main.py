import ctypes
import os
from functools import partial

from plotting import plot_results
from profile_models import (
    analyze_model,
)
from strategies import (
    greedy_by_size,
    bump_allocator,
    solve_cp,
    calculate_high_watermark,
    gergov,
    mincost_flow,
    greedy_by_longest,
    find_gap,
    GapPriority,
)


def plan_greedy_strats(req_mem_allocs, name):
    # planned_allocs = greedy_by_size(req_mem_allocs)
    # print("greedy_by_size_with_smallest_gap valid", verify_allocation(planned_allocs), flush=True)
    # print(f"{name}", "Greedy by Size", calculate_high_watermark(planned_allocs), sep=",", flush=True)
    # make_memory_map(planned_allocs, f"{name}.greedy_by_size_with_smallest_gap")

    planned_allocs = greedy_by_size(
        req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    )
    print(
        f"{name}",
        "greedy_by_size_with_first_gap",
        calculate_high_watermark(planned_allocs),
        sep=",",
        flush=True,
    )
    # make_memory_map(planned_allocs, f"{name}.greedy_by_size_with_first_gap")

    # planned_allocs = greedy_by_longest(req_mem_allocs)
    # print(f"{name}", "Greedy by Longest", calculate_high_watermark(planned_allocs), sep=",", flush=True)
    # make_memory_map(planned_allocs, f"{name}.greedy_by_longest_with_smallest_gap")

    planned_allocs = greedy_by_longest(
        req_mem_allocs, gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST)
    )
    print(
        f"{name}",
        "greedy_by_longest_with_first_gap",
        calculate_high_watermark(planned_allocs),
        sep=",",
        flush=True,
    )
    # make_memory_map(planned_allocs, f"{name}.greedy_by_longest_with_first_gap")


def plan_integer_programming_starts(req_mem_allocs, name):
    planned_allocs = solve_cp(req_mem_allocs)
    # print("solve_cp valid", verify_allocation(planned_allocs), flush=True)
    # print(f"{name}", "Symbolic CSP", calculate_high_watermark(planned_allocs), sep=",", flush=True)
    print(
        f"{name}",
        "Symbolic MIP",
        calculate_high_watermark(planned_allocs),
        sep=",",
        flush=True,
    )
    # make_memory_map(planned_allocs, f"{name}.solve_cp")

    # planned_allocs = solve_mip(req_mem_allocs)
    # # print("solve_mip valid", verify_allocation(planned_allocs), flush=True)
    # print(f"{name}", "Symbolic MIP", calculate_high_watermark(planned_allocs), sep=",", flush=True)
    # make_memory_map(planned_allocs, f"{name}.solve_mip")


def plan_other(req_mem_allocs, mem_events, name):
    planned_allocs = bump_allocator(mem_events)
    # print("bump valid", verify_allocation(planned_allocs), flush=True)
    print(
        f"{name}",
        "Bump Allocator",
        calculate_high_watermark(planned_allocs),
        sep=",",
        flush=True,
    )
    # make_memory_map(planned_allocs, f"{name}.bump")

    planned_allocs = mincost_flow(req_mem_allocs)
    # print("mincost_flow valid", verify_allocation(planned_allocs), flush=True)
    print(
        f"{name}",
        "Min-cost Flow",
        calculate_high_watermark(planned_allocs),
        sep=",",
        flush=True,
    )
    # make_memory_map(planned_allocs, f"{name}.mincost_flow")

    planned_allocs = gergov(req_mem_allocs)
    # print("gergov valid", verify_allocation(planned_allocs), flush=True)
    print(
        f"{name}",
        "Gergov",
        calculate_high_watermark(planned_allocs),
        sep=",",
        flush=True,
    )
    # make_memory_map(planned_allocs, f"{name}.gergov")


def make_maps(model, x, name):
    # trace_json = json.load(open(os.getcwd() + f"/traces/{name}.json"))
    # req_mem_allocs = get_required_mem_allocs(trace_json)
    # mem_events = get_mem_events(trace_json)

    req_mem_allocs, mem_events = analyze_model(model, x)
    # print("len req_mem_allocs", len(req_mem_allocs))

    plan_greedy_strats(req_mem_allocs, name)
    plan_integer_programming_starts(req_mem_allocs, name)
    plan_other(req_mem_allocs, mem_events, name)


if __name__ == "__main__":
    for dir in ["models", "memory_maps", "traces"]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    # with open('res.csv', 'w') as f:
    #     with redirect_stdout(f):
    #         print("model,dims,strategy,size")
    #         for name, model in vision_models():
    #             print(name, file=sys.stderr)
    #             if name in {"small_bert", "large_bert", "attention_is_all_you_need"}:
    #                 inp = make_bert_input()
    #             else:
    #                 inp = torch.rand((1, 3, 244, 244))
    #             try:
    #                 make_maps(model, inp, f"{name},1x3x244x244")
    #             except Exception as e:
    #                 print(name, e, file=sys.stderr)
    plot_results("res.csv")
    # hook_alloc_cpu()
