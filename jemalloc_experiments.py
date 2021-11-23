import argparse
import ast
import os
from collections import defaultdict
from multiprocessing import Process
from threading import Thread
from typing import Dict

import jemalloc_bindings


def test_je_malloc(N):
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
    return stats

def make_req_mem_allocs(allocs: Dict, frees: Dict, idx_ts):
    ptrs = {}
    for i, ts in idx_ts:
        for (alloc_sz, ptr_addr) in allocs.get(ts, []):
            real_ptr = jemalloc_bindings.je_malloc(alloc_sz)
            ptrs[ptr_addr] = real_ptr
        for (free_sz, ptr_addr) in frees.get(ts, []):
            real_ptr = ptrs[ptr_addr]
            jemalloc_bindings.je_free(real_ptr)


def make_lots_of_reqs(allocs: Dict, frees: Dict, idx_ts, num_loops):
    for i in range(num_loops):
        make_req_mem_allocs(allocs, frees, idx_ts)


def record_lots_of_reqs(name, num_workers, num_loops, threaded):
    allocs, frees, idx_ts = parse_reqs(
        f"/home/mlevental/dev_projects/pytorch_memory_planning/req_allocs/{name}.txt"
    )
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

    dir = f"je_malloc_runs/{name}/multi_worker"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    jemalloc_bindings.je_malloc_print_stats(
        f"je_malloc_runs/{name}/{num_workers}_{num_loops}.json"
    )


def heap_profile(name):
    allocs, frees, idx_ts = parse_reqs(
        f"/home/mlevental/dev_projects/pytorch_memory_planning/req_allocs/{name}.txt"
    )
    dir = f"je_malloc_runs/{name}/"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    stats = make_req_mem_allocs_with_print_stats(allocs, frees, idx_ts)
    with open(f"{dir}/heap_profile.csv", "w") as csv:
        csv.write("allocated,active,metadata,resident,mapped,retained\n")
        csv.write('\n'.join(stats) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("name", type=str)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_loops", type=int, default=1)
    parser.add_argument("--threaded", action="store_true", default=False)
    args = parser.parse_args()
    name = args.name
    num_workers = args.num_workers
    num_loops = args.num_loops
    threaded = args.threaded

    heap_profile(name)
    # record_lots_of_reqs(name, num_workers, num_loops, threaded)
