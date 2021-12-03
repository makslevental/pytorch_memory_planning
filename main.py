import glob
import multiprocessing
import os
import time
from functools import partial
from itertools import chain
from typing import List

from memory_observer import (
    load_ops_yaml,
    export_memory_reqs,
    get_call_chains,
    parse_ops_dict,
    with_log,
)
from profile_models import (
    analyze_model,
)
from strategies import (
    greedy_by_size,
    bump_allocator,
    solve_csp,
    greedy_by_longest,
    find_gap,
    GapPriority,
    verify_allocation,
    PlannedAlloc, mincost_flow, gergov,
)


class Timer:
    def __init__(self, model, strategy):
        self.model = model
        self.strategy = strategy

    def __enter__(self):
        self.start = time.monotonic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.monotonic()
        print(self.model, self.strategy, end - self.start, flush=True)


def plan_greedy_strats(req_mem_allocs, model_name):
    with Timer(model_name, "greedy_by_size_smallest_gap") as _:
        greedy_by_size_smallest_gap = greedy_by_size(req_mem_allocs)
    with Timer(model_name, "greedy_by_size_first_gap") as _:
        greedy_by_size_first_gap = greedy_by_size(req_mem_allocs,
                                                  gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST))
    with Timer(model_name, "greedy_by_longest_smallest_gap") as _:
        greedy_by_longest_smallest_gap = greedy_by_longest(req_mem_allocs)
    with Timer(model_name, "greedy_by_longest_first_gap") as _:
        greedy_by_longest_first_gap = greedy_by_longest(req_mem_allocs,
                                                        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST))

    return {
        "greedy_by_size_smallest_gap": greedy_by_size_smallest_gap,
        "greedy_by_size_first_gap": greedy_by_size_first_gap,
        "greedy_by_longest_smallest_gap": greedy_by_longest_smallest_gap,
        "greedy_by_longest_first_gap": greedy_by_longest_first_gap,
    }


def plan_programming_starts(req_mem_allocs, model_name):
    with Timer(model_name, "csp"):
        csp = solve_csp(req_mem_allocs)

    return {
        "csp": csp,
        # "mip": solve_mip(req_mem_allocs),
    }


def plan_other_strats(req_mem_allocs, model_name):
    with Timer(model_name, "bump"):
        bump = bump_allocator(req_mem_allocs)
    with Timer(model_name, "mincost_flow"):
        flow = mincost_flow(req_mem_allocs)
    with Timer(model_name, "gergov"):
        gerg = gergov(req_mem_allocs)
    return {
        "bump": bump,
        "mincost_flow": flow,
        "gergov": gerg,
    }


def make_maps(model, x, name):
    # trace_json = json.load(open(os.getcwd() + f"/traces/{name}.json"))
    # req_mem_allocs = get_required_mem_allocs(trace_json)
    # mem_events = get_mem_events(trace_json)

    req_mem_allocs, mem_events = analyze_model(model, x)
    # print("len req_mem_allocs", len(req_mem_allocs))

    plan_greedy_strats(req_mem_allocs, name)
    plan_programming_starts(req_mem_allocs, name)
    plan_other_strats(req_mem_allocs, name)


def maybe_make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def make_planned_allocs_csv(model_name, params):
    ops_dict = load_ops_yaml(f"{model_name}.{params}")
    ops_dict = parse_ops_dict(ops_dict)
    _, _, names_to_allocs = get_call_chains(ops_dict)
    lvr_to_req_plus_root_caller = export_memory_reqs(ops_dict, names_to_allocs)
    req_allocs = [req for _lvr, (req, _fn_name) in lvr_to_req_plus_root_caller.items()]
    print(model_name, "num allocs", len(req_allocs), flush=True)

    strats = []
    strats.append(plan_greedy_strats(req_allocs, model_name))
    if len(req_allocs) < 10000:
        strats.append(plan_programming_starts(req_allocs, model_name))
    strats.append(plan_other_strats(req_allocs, model_name))

    for strat, plan in chain(*[s.items() for s in strats]):
        plan: List[PlannedAlloc]
        plan.sort(key=lambda p: p.lvr.begin)
        if not verify_allocation(plan):
            print(f"{model_name} {strat} memory plan not valid")
        else:
            maybe_make_dir(f"planned_allocs/{model_name}")
            with open(f"planned_allocs/{model_name}/{strat}.{params}.csv", "w") as f:
                for p_alloc in plan:
                    root_caller = lvr_to_req_plus_root_caller[
                        p_alloc.lvr
                    ].root_caller_fn_name
                    f.writelines(
                        [
                            f"{p_alloc.lvr.begin},{p_alloc.lvr.end},{p_alloc.mem_region.offset},{p_alloc.mem_region.size},{root_caller}\n"
                        ]
                    )


if __name__ == "__main__":
    for dir in [
        "benchmarks",
        "call_chains",
        "je_malloc_runs",
        "jemalloc_plots",
        "logs" "memory_maps",
        "models",
        "perf_records",
        "planned_allocs",
        "profiles",
        "req_allocs",
        "symbolic_models",
    ]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for i, fp in enumerate(glob.glob("profiles/*.yml")):
            _, fn = os.path.split(fp)
            model_name, params, _yml = fn.split(".")
            # make_planned_allocs_csv(model_name, params)
            pool.apply_async(with_log, (make_planned_allocs_csv, (model_name, params)))
        pool.close()
        pool.join()
