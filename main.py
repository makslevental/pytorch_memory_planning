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
import pandas as pd

from memory_observer import (
    load_ops_yaml,
    export_memory_reqs,
    parse_ops_dict,
    get_call_chains,
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
        print(
            ",".join(list(map(str, [self.model, self.strategy, end - self.start]))),
            flush=True,
        )


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
        memory_planning_cpp(
            req_mem_allocs_df, memory_planning.BUMP
        )
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


def make_planned_allocs_csv(model_name, params):
    ops_dict = load_ops_yaml(model_name, params)
    ops_dict = parse_ops_dict(ops_dict)
    _, _, names_to_allocs = get_call_chains(ops_dict)
    lvr_to_req_plus_root_caller = export_memory_reqs(ops_dict, names_to_allocs)
    req_allocs = [req for _lvr, (req, _fn_name) in lvr_to_req_plus_root_caller.items()]
    req_allocs_df = make_df_from_reqs(req_allocs)
    print(
        ",".join(map(str, [model_name, params, "num allocs", len(req_allocs_df)])),
        flush=True,
    )

    strats = []
    if not os.path.exists(
            f"planned_allocs/{model_name}/{params}/greedy_by_size_first_gap.csv"
    ):
        strats.append(plan_greedy_strats(req_allocs_df, model_name + "." + params))
    if not os.path.exists(f"planned_allocs/{model_name}/{params}/bump.csv"):
        strats.append(
            plan_other_strats(req_allocs_df, model_name + "." + params, req_allocs)
        )
    if not os.path.exists(f"planned_allocs/{model_name}/{params}/csp.csv"):
        if len(req_allocs_df) < 10000:
            strats.append(
                plan_programming_strats(req_allocs_df, model_name + "." + params)
            )

    for strat, plan in chain(*[s.items() for s in strats]):
        plan: List[PlannedAlloc]
        plan.sort(key=lambda p: p.lvr.begin)
        if not verify_allocation(plan):
            print(f"{model_name}.{params} {strat} memory plan not valid", file=sys.stderr, flush=True)
        else:
            maybe_make_dir(f"planned_allocs/{model_name}")
            maybe_make_dir(f"planned_allocs/{model_name}/{params}")
            fp = f"planned_allocs/{model_name}/{params}/{strat}.csv"
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


TWOS = [0, 3, 5]

NUM_THREADS = str(multiprocessing.cpu_count())
NUM_THREADS = "1"


def get_all_names(batch_sizes=[1]):
    names = list(map(lambda x: x.strip(), open("important_models.txt").readlines()))
    hws = [32, 64, 128, 256, 512]

    for batch_size in batch_sizes:
        for hw in hws:
            for model_name in names:
                if "inception" in model_name and hw < 128:
                    continue
                if "alexnet" in model_name and hw < 64:
                    continue
                if "dcgan" in model_name and hw < 64:
                    continue
                yield model_name, batch_size, hw


def run_all_mem_experiments(
        bin_path,
):
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_DEBUG_CPU_TYPE": "5",
        # "MEMORY_DEBUG": "false",
    }
    print(" ".join([f"{k}={v}" for k, v in env.items()]))
    names = list(get_all_names([1, 32, 64]))
    num_repeats = 1
    num_warmup = 10
    nump_loops = 30
    strat = "csp"
    with open(f"memory_run_times.csv", "w", buffering=1) as time_log, open(
            f"err.log", "w", buffering=1
    ) as err_log:
        time_log.write(
            "je_or_me,model_name,num_workers,batch_size,hw,num_total_iters,total,ms_per_iter\n"
        )
        for i, (model_name, batch_size, hw) in enumerate(names):
            for num_workers in [1, 32, 64]:
                for je_or_me in ["je", "me"]:
                    model_fp = f"{os.getcwd()}/models/{model_name}.x1.y{hw}.pt"
                    plan_fp = f"{os.getcwd()}/planned_allocs/{model_name}/x1.y{hw}/{strat}.csv"

                    if not os.path.exists(plan_fp):
                        plan_fp = f"{os.getcwd()}/planned_allocs/{model_name}/x1.y{hw}/greedy_by_size_first_gap.csv"
                        if not os.path.exists(plan_fp):
                            print(plan_fp, "doesn't exist wtf", file=sys.stderr)
                            continue

                    cmd = [bin_path]
                    params = list(
                        map(
                            str,
                            [
                                je_or_me,
                                model_name,
                                strat,
                                num_workers,
                                num_repeats,
                                num_warmup,
                                nump_loops,
                                batch_size,
                                hw,
                                model_fp,
                                plan_fp,
                            ],
                        )
                    )
                    cmd.extend(params)
                    print(" ".join(cmd), flush=True)
                    df = pd.read_csv(plan_fp, names=["begin", "end", "offset", "size", "root_coller"])
                    peak_usage = (df["offset"] + df["size"]).max() * batch_size * num_workers
                    if peak_usage > 100 << 30:  # 100GB
                        print(plan_fp, "will hit OOM", file=sys.stderr)
                        continue
                       
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    try:
                        outs, errs = proc.communicate()
                        outs = outs.decode().strip()
                        print(outs, flush=True)
                        time_log.write(f"{outs}\n")
                        time_log.flush()
                    except Exception as e:
                        proc.kill()
                        outs, errs = proc.communicate()
                        outs = outs.decode().strip()
                        errs = errs.decode().strip()

                        print(errs, file=sys.stderr, flush=True)
                        err_log.write(f"{cmd}; errs: {errs} outs: {outs}")
                        err_log.flush()


def make_one_profile(bin_path, env, model_name, batch_size, hw):
    cmd = [bin_path]
    model_fp = f"{os.getcwd()}/models/{model_name}.x1.y{hw}.pt"
    params = list(
        map(
            str,
            ["NONE", model_name, "NONE", 0, 0, 0, 0, batch_size, hw, model_fp, "NONE"],
        )
    )
    cmd.extend(params)
    print(" ".join(cmd))

    with open(
            f"logs/{model_name}.x{batch_size}.y{hw}.err.log", "w", buffering=1
    ) as err_log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            outs, errs = proc.communicate()
            outs = outs.decode().strip()
        except Exception as e:
            proc.kill()
            outs, errs = proc.communicate()
            outs = outs.decode().strip()
            errs = errs.decode().strip()

            print(errs, file=sys.stderr)
            err_log.write(f"{cmd}; errs: {errs} outs: {outs}")
            err_log.flush()

    with open(
            f"profiles/{model_name}.x{batch_size}.y{hw}.yml", "w", buffering=1
    ) as profile:
        profile.write(f"{outs}\n")
        profile.flush()


def make_profiles(bin_path):
    env = {
        "OPENBLAS_NUM_THREADS": NUM_THREADS,
        "GOTO_NUM_THREADS": NUM_THREADS,
        "OMP_NUM_THREADS": NUM_THREADS,
        "MKL_NUM_THREADS": NUM_THREADS,
        "MKL_DEBUG_CPU_TYPE": "5",
        "MEMORY_DEBUG": "true",
    }

    with multiprocessing.Pool(multiprocessing.cpu_count() // 8) as pool:
        for model_name, batch_size, hw in get_all_names():
            fp = f"profiles/{model_name}.x{batch_size}.y{hw}.yml"
            if os.path.exists(fp):
                continue
            # make_one_profile(bin_path, env, model_name, batch_size, hw)
            pool.apply_async(
                make_one_profile, (bin_path, env, model_name, batch_size, hw)
            )
        pool.close()
        pool.join()


def make_all_planned_allocs():
    with multiprocessing.Pool(multiprocessing.cpu_count() // 4) as pool:
        for model_name, batch_size, hw in get_all_names():
            # make_planned_allocs_csv(model_name,  f"x{batch_size}.y{hw}")
            pool.apply_async(
                make_planned_allocs_csv, (model_name, f"x{batch_size}.y{hw}")
            )
        pool.close()
        pool.join()


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

    # make_profiles("/home/mlevental/dev_projects/pytorch_shape_inference/cmake-build-debug/bin/pytorch_memory_allocator")
    # make_all_planned_allocs()
    run_all_mem_experiments(
        "/home/mlevental/dev_projects/pytorch_shape_inference/cmake-build-debug/bin/pytorch_memory_allocator"
    )
