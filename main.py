import glob
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from functools import partial
from itertools import chain
from typing import List

import memory_planning
from numpy import mean

from memory_observer import (
    load_ops_yaml,
    export_memory_reqs,
    get_call_chains,
    parse_ops_dict,
)
from strategies import (
    greedy_by_size,
    bump_allocator,
    solve_csp,
    greedy_by_longest,
    find_gap,
    GapPriority,
    verify_allocation,
    PlannedAlloc,
    gergov,
    make_df_from_reqs,
    solve_mip,
    ortools_mincost_flow,
    memory_planning_cpp,
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


def plan_greedy_strats(req_mem_allocs_df, model_name):
    with Timer(model_name, "greedy_by_size_smallest_gap") as _:
        memory_planning_cpp(
            req_mem_allocs_df, memory_planning.GREEDY_BY_SIZE_WITH_SMALLEST_GAP
        )
    greedy_by_size_smallest_gap = greedy_by_size(req_mem_allocs_df)
    with Timer(model_name, "greedy_by_size_first_gap") as _:
        memory_planning_cpp(
            req_mem_allocs_df, memory_planning.GREEDY_BY_SIZE_WITH_FIRST_GAP
        )
    greedy_by_size_first_gap = greedy_by_size(
        req_mem_allocs_df,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST),
    )
    with Timer(model_name, "greedy_by_longest_smallest_gap") as _:
        memory_planning_cpp(
            req_mem_allocs_df,
            memory_planning.GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP,
        )
    greedy_by_longest_smallest_gap = greedy_by_longest(req_mem_allocs_df)
    with Timer(model_name, "greedy_by_longest_first_gap") as _:
        memory_planning_cpp(
            req_mem_allocs_df, memory_planning.GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP
        )
    greedy_by_longest_first_gap = greedy_by_longest(
        req_mem_allocs_df,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST),
    )

    return {
        "greedy_by_size_smallest_gap": greedy_by_size_smallest_gap,
        "greedy_by_size_first_gap": greedy_by_size_first_gap,
        "greedy_by_longest_smallest_gap": greedy_by_longest_smallest_gap,
        "greedy_by_longest_first_gap": greedy_by_longest_first_gap,
    }


def plan_programming_strats(req_mem_allocs_df, model_name):
    with Timer(model_name, "csp"):
        csp = solve_csp(req_mem_allocs_df)
    # with Timer(model_name, "mip"):
    #     mip = solve_mip(req_mem_allocs_df)
    with Timer(model_name, "mincost_flow"):
        mincost_flow = ortools_mincost_flow(req_mem_allocs_df)

    return {"csp": csp, "mincost_flow": mincost_flow}


def plan_other_strats(req_mem_allocs_df, model_name, req_mem_allocs):
    with Timer(model_name, "bump"):
        bump = bump_allocator(req_mem_allocs)
    with Timer(model_name, "gergov"):
        memory_planning_cpp(req_mem_allocs_df, memory_planning.GERGOV)
    gerg = gergov(req_mem_allocs_df)
    return {
        "bump": bump,
        "gergov": gerg,
    }


def maybe_make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def save_requirements_df(req_allocs_df, model_name, params):
    req_allocs_df.reset_index()[["begin", "end", "mem_size"]].to_csv(
        f"reqs/{model_name}/{params}.csv", header=False
    )


def make_planned_allocs_csv(model_name, params=None, num_workers=1):
    ops_dict = load_ops_yaml(model_name, params)
    ops_dict = parse_ops_dict(ops_dict)
    _, _, names_to_allocs = get_call_chains(ops_dict)
    lvr_to_req_plus_root_caller = export_memory_reqs(ops_dict, names_to_allocs)
    req_allocs = [req for _lvr, (req, _fn_name) in lvr_to_req_plus_root_caller.items()]
    req_allocs_df = make_df_from_reqs(req_allocs)
    print(model_name, "num allocs", len(req_allocs_df), flush=True)

    strats = []
    strats.append(plan_greedy_strats(req_allocs_df, model_name))
    strats.append(plan_other_strats(req_allocs_df, model_name, req_allocs))
    if len(req_allocs_df) < 10000:
        strats.append(plan_programming_strats(req_allocs_df, model_name))

    for strat, plan in chain(*[s.items() for s in strats]):
        plan: List[PlannedAlloc]
        plan.sort(key=lambda p: p.lvr.begin)
        if not verify_allocation(plan):
            print(f"{model_name} {strat} memory plan not valid")
        else:
            maybe_make_dir(f"planned_allocs/{model_name}")
            if params is not None:
                maybe_make_dir(f"planned_allocs/{model_name}/{params}")
                fp = f"planned_allocs/{model_name}/{params}/{strat}.csv"
            else:
                fp = f"planned_allocs/{model_name}/{strat}.csv"
            with open(fp, "w") as f:
                for p_alloc in plan:
                    if p_alloc.lvr not in lvr_to_req_plus_root_caller:
                        raise "wtf"
                    root_caller = lvr_to_req_plus_root_caller[
                        p_alloc.lvr
                    ].root_caller_fn_name
                    f.writelines(
                        [
                            f"{p_alloc.lvr.begin},{p_alloc.lvr.end},{p_alloc.mem_region.offset},{p_alloc.mem_region.size},{root_caller}\n"
                        ]
                    )


def with_log(fn, args):
    sys.stderr = open(f"logs/{'.'.join(map(str, args))}.err", "w")
    # sys.stdout = open(f"logs/{'.'.join(args)}.log", "w")
    try:
        fn(*args)
    except:
        traceback.print_exc()


from subprocess import Popen


def run_cmd(cmd: List[str], out_pipe, err_pipe):
    return Popen(cmd, stdout=out_pipe, stderr=err_pipe)


def run_mem_experiments_per_model(
        time_log,
        err_log,
        je_or_me,
        bin_path,
        model_name,
        warmup=10,
        num_loops=10,
        num_repeats=1,
):
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_DEBUG_CPU_TYPE": "5",
    }

    for num_workers_exp in range(7):
        num_workers = 2 ** num_workers_exp
        for batch_exp in range(7):
            batch_size = 2 ** batch_exp
            for hw_exp in range(7):
                hw = 2 ** hw_exp
                if "inception" in model_name and hw < 128:
                    continue
                if "alexnet" in model_name and hw < 64:
                    continue
                if "dcgan" in model_name and hw < 64:
                    continue

                cmd = [bin_path]
                params = list(
                    map(
                        str,
                        [
                            je_or_me,
                            model_name,
                            "csp",
                            num_workers,
                            num_repeats,
                            warmup,
                            num_loops,
                            batch_size,
                            hw,
                        ],
                    )
                )
                cmd.extend(params)
                print(" ".join(cmd))

                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    outs, errs = proc.communicate()
                    outs = outs.decode().strip()
                    time_log.write(f"{outs}\n")
                    time_log.flush()
                except Exception as e:
                    proc.kill()
                    outs, errs = proc.communicate()
                    outs = outs.decode().strip()
                    errs = errs.decode().strip()

                    print(errs, file=sys.stderr)
                    err_log.write(f"{cmd}; errs: {errs} outs: {outs}")
                    err_log.flush()


def run_all_mem_experiments(
        bin_path,
):
    names = list(map(lambda x: x.strip(), open("model_names.txt").readlines()))
    with open(f"times.csv", "w", buffering=1) as time_log, open(
            f"err.log", "w", buffering=1
    ) as err_log:
        time_log.write("je_or_me,model_name,batch_size,hw,total,ms_per_iter\n")
        for je_or_me in ["me", "je"]:
            for model_name in names:
                print(model_name)
                if "alex" not in model_name:
                    continue
                run_mem_experiments_per_model(
                    time_log, err_log, je_or_me, bin_path, model_name
                )


if __name__ == "__main__":
    for dir in [
        "benchmarks",
        "call_chains",
        "je_malloc_runs",
        "jemalloc_plots",
        "logs",
        "memory_maps",
        "models",
        "perf_records",
        "planned_allocs",
        "profiles",
        "req_allocs",
        "reqs",
        "symbolic_models",
    ]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    run_all_mem_experiments(
        "/home/mlevental/dev_projects/pytorch_shape_inference/cmake-build-debug-clang/bin/pytorch_memory_allocator"
    )

    # for i, fp in enumerate(glob.glob("profiles/*.yml")):
    #     _, fn = os.path.split(fp)
    #
    # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    #     for i, fp in enumerate(glob.glob("profiles/*.yml")):
    #         _, fn = os.path.split(fp)
    #         model_name, params = os.path.splitext(fn)[0].split(".", 1)
    #         # make_planned_allocs_csv(model_name, params)
    #         # pool.apply_async(with_log, (make_planned_allocs_csv, (model_name, params)))
    #         pool.apply_async(make_planned_allocs_csv, (model_name, params))
    #     pool.close()
    #     pool.join()
