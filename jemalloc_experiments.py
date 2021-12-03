import argparse
import ast
import os
from collections import defaultdict
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import Dict

import torch


def test_je_malloc(N):
    import jemalloc_bindings

    allocs = []
    for i in range(N):
        allocs.append(jemalloc_bindings.je_malloc(i))
    for a in allocs:
        jemalloc_bindings.je_free(a)

    jemalloc_bindings.je_malloc_print_stats(f"je_malloc_runs/test_je_malloc_{N}.json")


def parse_reqs(fp):
    f = open(fp, "r").read()
    lvrs = ast.literal_eval(f)

    allocs = defaultdict(list)
    frees = defaultdict(list)
    for (b, e), (sz, ptr_addr) in lvrs.items():
        allocs[b].append((sz, ptr_addr))
        frees[e].append((sz, ptr_addr))

    assert sum(len(l) for l in allocs.values()) == sum(len(l) for l in frees.values())

    unique_tses = sorted(
        list(set([b for b, e in lvrs.keys()] + [e + 1 for b, e in lvrs.keys()]))
    )
    idx_ts = list(enumerate(unique_tses))
    return allocs, frees, idx_ts


def make_req_mem_allocs_with_print_stats(allocs: Dict, frees: Dict, idx_ts):
    import jemalloc_bindings

    stats = []
    ptrs = {}
    for i, ts in idx_ts:
        for (alloc_sz, ptr_addr) in allocs.get(ts, []):
            real_ptr = jemalloc_bindings.je_malloc(alloc_sz)
            ptrs[ptr_addr] = real_ptr
        for (free_sz, ptr_addr) in frees.get(ts, []):
            real_ptr = ptrs[ptr_addr]
            jemalloc_bindings.je_free(real_ptr)
        stats.append(jemalloc_bindings.get_stats())
        # t = randint(1, 10) / 1000
        # print(t)
        # sleep(t)
    return stats


def make_req_mem_allocs(allocs: Dict, frees: Dict, idx_ts):
    import jemalloc_bindings

    ptrs = {}
    for i, ts in idx_ts:
        for (alloc_sz, ptr_addr) in allocs.get(ts, []):
            real_ptr = jemalloc_bindings.je_malloc(alloc_sz)
            ptrs[ptr_addr] = real_ptr
        for (free_sz, ptr_addr) in frees.get(ts, []):
            real_ptr = ptrs[ptr_addr]
            jemalloc_bindings.je_free(real_ptr)
        # t = randint(1, 10) / 1000
        # print(t)
        # sleep(t)


def make_lots_of_reqs(allocs: Dict, frees: Dict, idx_ts, num_loops):
    for i in range(num_loops):
        make_req_mem_allocs(allocs, frees, idx_ts)


def print_je_malloc_stats(name):
    import jemalloc_bindings

    dir = f"je_malloc_runs/{name}/"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    jemalloc_bindings.je_malloc_print_stats(
        f"je_malloc_runs/{name}/{narenas}_{num_workers}_{num_loops}.json"
    )


def set_num_background_threads(n):
    import jemalloc_bindings

    jemalloc_bindings.set_num_background_threads(n)


def record_lots_of_reqs(name, num_workers, num_loops, threaded):
    allocs, frees, idx_ts = parse_reqs(f"req_allocs/{name}.txt")
    workers = []
    for i in range(num_workers):
        if threaded:
            worker = Thread(
                target=make_lots_of_reqs, args=(allocs, frees, idx_ts, num_loops)
            )
        else:
            worker = Process(
                target=make_lots_of_reqs, args=(allocs, frees, idx_ts, num_loops)
            )
        workers.append(worker)
        worker.start()

    for t in workers:
        t.join()

    print_je_malloc_stats(name)


def heap_profile(name):
    allocs, frees, idx_ts = parse_reqs(f"req_allocs/{name}.txt")
    dir = f"je_malloc_runs/{name}/"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    stats = []
    for i in range(100):
        make_req_mem_allocs_with_print_stats(allocs, frees, idx_ts)
    stats.extend(make_req_mem_allocs_with_print_stats(allocs, frees, idx_ts))
    with open(f"{dir}/heap_profile.csv", "w") as csv:
        csv.write("allocated,active,metadata,resident,mapped,retained\n")
        csv.write("\n".join(stats) + "\n")


def torch_jemalloc(name, worker):
    # import psutil
    # p = psutil.Process()
    # print(f"Child #{worker}: {p}, affinity {p.cpu_affinity()}", flush=True)
    # p.cpu_affinity([worker])
    with torch.no_grad():
        model = torch.jit.load(f"models/{name}.pt")
        x = torch.rand((1, 3, 1024, 1024))
        for _ in range(num_loops):
            model(x)


def narena_torch_jemalloc(name, num_workers):
    with ThreadPool(processes=num_workers) as pool:
        workers: int = num_workers
        for i in range(workers):
            pool.apply_async(torch_jemalloc, (name, i))

        pool.close()
        pool.join()

    print_je_malloc_stats(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("name", type=str)
    parser.add_argument("--narenas", type=str, default=257)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_loops", type=int, default=1)
    parser.add_argument("--threaded", action="store_true", default=False)
    parser.add_argument("--heap_profile", action="store_true", default=False)

    args = parser.parse_args()
    name = args.name
    narenas = args.narenas
    num_workers = args.num_workers
    num_loops = args.num_loops
    threaded = args.threaded
    do_heap_profile = args.heap_profile

    model_pts = set()
    for (dirpath, dirnames, filenames) in os.walk("models"):
        model_pts.update([f.replace(".pt", "") for f in filenames])
        break

    if name in model_pts:
        print(
            f"{name=}, {narenas=}, {num_workers=}, {num_loops=}, {threaded=}, {do_heap_profile=}"
        )
        narena_torch_jemalloc(name, num_workers)
    # model = vision_models(name).eval()
    # main()
    # if do_heap_profile:
    #     heap_profile(name)
    # else:
    #     record_lots_of_reqs(name, num_workers, num_loops, threaded)
