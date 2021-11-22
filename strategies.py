import json
import sys
from collections import namedtuple, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from functools import partial, total_ordering
from pprint import pprint
from typing import List, Dict

import networkx as nx
import numpy as np
from intervaltree import IntervalTree
from ncls import NCLS
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from z3 import Optimize, Bools, Not, And


def make_ncls_from_tuples(tups):
    return NCLS(
        np.array([t for (t, _, _) in tups], dtype=np.int64),
        np.array([t for (_, t, _) in tups], dtype=np.int64),
        np.array([t for (_, _, t) in tups], dtype=np.int64),
    )


def valid_add(a, b):
    return a + b >= a


def valid_sub(a, b):
    return a >= b


def overlap(a, b, c, d):
    assert a <= b
    assert c <= d

    outer_len = max(b, d) - min(a, c)
    interval_len_1 = b - a
    interval_len_2 = d - c

    if not valid_add(interval_len_1, interval_len_2) or not valid_sub(
        outer_len, interval_len_1 + interval_len_2
    ):
        return True
    else:
        return False


@total_ordering
@dataclass
class LiveRange:
    begin: int
    end: int

    def __eq__(self, other):
        return self.begin == other.begin and self.end == other.end

    def __lt__(self, other):
        return self.begin < other.begin and self.end < other.begin

    def __len__(self):
        return self.end - self.begin + 1

    def overlap(self, other):
        return overlap(self.begin, self.end + 1, other.begin, other.end + 1)

    def __str__(self):
        return f"{(self.begin, self.end)}"


@total_ordering
@dataclass
class MemRegion:
    offset: int
    size: int

    def __eq__(self, other):
        return self.offset == other.begin and self.size == other.end

    def __lt__(self, other):
        return self.offset < other.begin and self.size < other.begin

    def __len__(self):
        return self.size - self.offset + 1

    def overlap(self, other):
        return overlap(
            self.offset, self.next_free_addr, other.offset, other.next_free_addr
        )

    @property
    def next_free_addr(self):
        return self.offset + self.size

    def __str__(self):
        return f"{(self.offset, self.size)}"


import hashlib


@dataclass
class RequiredAlloc:
    lvr: LiveRange
    size: int
    ptr_addr: str

    def __str__(self):
        return f"{self.lvr}:{self.size}"

    def __hash__(self):
        return int(hashlib.sha256(str(self).encode("utf-8")).hexdigest(), 16) % 10 ** 8


@dataclass
class PlannedAlloc:
    lvr: LiveRange
    mem_region: MemRegion

    @classmethod
    def from_req(cls, req_alloc: RequiredAlloc, offset: int):
        return PlannedAlloc(req_alloc.lvr, MemRegion(offset, req_alloc.size))

    def __str__(self):
        return f"{self.lvr}:{self.mem_region}"

    def __repr__(self):
        return str(self)

    def overlap(self, other):
        return self.lvr.overlap(other.lvr) and self.mem_region.overlap(other.mem_region)


class GapPriority(Enum):
    SMALLEST = 1
    FIRST = 2


def find_gap(
    record: RequiredAlloc,
    current_allocs: Dict[str, PlannedAlloc],
    interval_tree: NCLS,
    *,
    GAP_PRIORITY: GapPriority = GapPriority.SMALLEST,
):
    best_gap = float("inf")
    best_offset = None
    prev_offset = 0

    overlapping_sorted_allocs = [
        current_allocs[interval.data]
        for interval in interval_tree.overlap(record.lvr.begin, record.lvr.end + 1)
        if interval.data in current_allocs and interval.data != record.ptr_addr
    ]
    overlapping_sorted_allocs.sort(key=lambda x: x.mem_region.offset)
    for alloc in overlapping_sorted_allocs:
        # offset_x will be ahead of the previous block
        # while prev_offset will be just in front
        # this looks for small gap ahead of a block
        gap = alloc.mem_region.offset - prev_offset
        if record.size <= gap < best_gap:
            best_offset = prev_offset
            if GAP_PRIORITY == GapPriority.FIRST:
                break
            best_gap = gap

        prev_offset = max(prev_offset, alloc.mem_region.next_free_addr)

    if best_offset is None:
        best_offset = prev_offset
    return best_offset


def _greedy_by_size(
    sorted_req_mem_allocs: List[RequiredAlloc],
    gap_finder,
):
    current_allocs: Dict[str, PlannedAlloc] = {}
    interval_tree = IntervalTree.from_tuples(
        (req.lvr.begin, req.lvr.end + 1, req.ptr_addr) for req in sorted_req_mem_allocs
    )
    inorder_of_decision_allocs: List[PlannedAlloc] = []
    for mem_alloc in sorted_req_mem_allocs:
        best_offset = gap_finder(mem_alloc, current_allocs, interval_tree)

        p = PlannedAlloc.from_req(mem_alloc, best_offset)
        inorder_of_decision_allocs.append(p)
        current_allocs[mem_alloc.ptr_addr] = p

    return inorder_of_decision_allocs


def greedy_by_size(
    req_mem_allocs: List[RequiredAlloc],
    *,
    gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
):
    print("greedy by size", file=sys.stderr)
    # biggest size first but break ties deterministically
    req_mem_allocs.sort(key=lambda r: (r.size, r.lvr), reverse=True)
    return _greedy_by_size(req_mem_allocs, gap_finder)


def greedy_by_longest(
    req_mem_allocs: List[RequiredAlloc],
    *,
    gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
):
    print("greedy by longest", file=sys.stderr)
    req_mem_allocs.sort(key=lambda r: (len(r.lvr), r.lvr), reverse=True)
    return _greedy_by_size(req_mem_allocs, gap_finder)


def save_planned_allocs(allocs: List[PlannedAlloc], name):
    json.dump(
        [str(a) for a in allocs], open(f"planned_allocs/{name}", "w"), indent=True
    )


MemEvent = namedtuple("MemEvent", "ptr_addr size ts")


def solve_z3():
    a, b, c = Bools("a b c")
    o = Optimize()
    o.add(a == c)
    o.add(Not(And(a, b)))
    o.add_soft(a, 2)
    o.add_soft(b, 3)
    o.add_soft(c, 1)
    print(o.check())
    print(o.model())


def bump_allocator(mem_events: List[MemEvent]):
    print("bump_allocator", file=sys.stderr)
    mem_events.sort(key=lambda r: r.ts)
    planned_allocations: Dict[str, PlannedAlloc] = {}
    next_offset = 0
    curr_allocs = 0
    for ptr_addr, size, ts in mem_events:
        if size > 0:
            planned_allocations[ptr_addr] = PlannedAlloc(
                LiveRange(ts, -1), MemRegion(next_offset, size)
            )
            next_offset += size
            curr_allocs += 1
        elif size < 0:
            assert ptr_addr in planned_allocations
            assert planned_allocations[ptr_addr].mem_region.size == -size
            assert planned_allocations[ptr_addr].lvr.begin < ts
            planned_allocations[ptr_addr].lvr.end = ts
            curr_allocs -= 1
        if curr_allocs == 0:
            next_offset = 0

    return list(planned_allocations.values())


def first_available(color_list):
    """Return smallest non-negative integer not in the given list of colors."""
    color_set = set(color_list)
    count = 0
    while True:
        if count not in color_set:
            return count
        count += 1


def greedy_color(G, order):
    color = dict()
    for node in order:
        used_neighbour_colors = [color[nbr] for nbr in G[node] if nbr in color]
        color[node] = first_available(used_neighbour_colors)
    return color


def gergov(required_allocs: List[RequiredAlloc]):
    print("gergov", file=sys.stderr)
    H = {
        (
            min([a.lvr.begin for a in required_allocs]),
            max([a.lvr.end for a in required_allocs]) + 1,
            0,
        )
    }
    J = [(r.lvr.begin, r.lvr.end + 1, r.size) for r in required_allocs]

    def if_there_exists(xl, xr):
        HH = IntervalTree.from_tuples(H)
        # JJ = IntervalTree.from_tuples(J)
        JJ = IntervalTree.from_tuples(J)
        for (r, c, s) in sorted(JJ.overlap(xl, xr), key=lambda j: j[0]):
            # for (r, c, s) in JJ.overlap(xl, xr):
            # for (r, c, s) in JJ.overlap(xl, xr):
            if not HH.overlaps(r, c):
                return (r, c, s)

        return None

    alphap = {}
    V = []
    while J:
        pick = (xl, xr, w) = min(H, key=lambda h: h[2])
        H.remove(pick)
        j = if_there_exists(xl, xr)
        if j is not None:
            r, c, s = j

            J.remove((r, c, s))
            V.append((r, c, s))
            alphap[(r, c, s)] = w

            H.add((max(xl, r), min(c, xr), w + s))
            if xl < r:
                H.add((xl, r, w))
            if c < xr:
                H.add((c, xr, w))

    # assert len(V) == len(required_allocs)
    # for v in V:
    #     assert v in [(r.size, r.lvr.begin, r.lvr.end+1) for r in required_allocs]
    # for r in required_allocs:
    #     assert (r.size, r.lvr.begin, r.lvr.end+1) in V
    VV = IntervalTree.from_tuples(V)
    E = defaultdict(list)
    for u in VV:
        ur, uc, us = u
        alphap_u = alphap[ur, uc, us]
        for v in VV.overlap(ur, uc):
            if u == v:
                continue
            vr, vc, vs = v

            alphap_v = alphap[vr, vc, vs]
            if overlap(alphap_u, alphap_u + us, alphap_v, alphap_v + vs):
                E[ur, uc, us].append((vr, vc, vs))

    alpha = {}
    color = greedy_color(E, V)
    max_j = max([alphap[rr, cc, ss] + ss for (rr, cc, ss) in V])
    for (r, c, s), cc in color.items():
        alpha[r, c, s] = alphap[r, c, s] + cc * max_j

    planned_allocs = []
    for (r, c, s), w in alpha.items():
        lvr = LiveRange(r, c - 1)
        mem_region = MemRegion(w, s)
        planned_allocs.append(PlannedAlloc(lvr, mem_region))
    return planned_allocs


def shortest_path_faster(E, W, s):
    d = {}
    for g in E:
        d[g] = float("inf")
    d[s] = 0
    Q = deque([s])
    while Q:
        u = Q.popleft()
        for v in E[u]:
            if d[u] + W[u, v] < d[v]:
                d[v] = d[u] + W[u, v]
                if v not in Q:
                    Q.append(v)


import matplotlib.pyplot as plt


def draw_nx(G):
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(
        G,
        pos,
        edge_color="black",
        width=1,
        linewidths=1,
        node_size=500,
        node_color="pink",
        alpha=0.9,
        labels={node: node for node in G.nodes()},
    )
    edge_labels = {}
    for u, vs in G.adj.items():
        for v in vs:
            edge_labels[u, v] = f"{vs[v]['weight']}, {vs[v]['capacity']}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.axis("off")
    plt.show()


def mincost_flow(required_allocs: List[RequiredAlloc]):
    required_allocs.sort(key=lambda r: r.lvr.begin)
    required_allocs_to_idx = {i: r for i, r in enumerate(required_allocs)}

    G = nx.DiGraph()
    G.add_node("s", demand=-len(required_allocs))
    G.add_node("t", demand=len(required_allocs))

    for i, req in enumerate(required_allocs):
        G.add_node(f"l,{i}", demand=0)
        G.add_node(f"r,{i}", demand=0)

        G.add_edge("s", f"r,{i}", weight=req.size, capacity=1)
        G.add_edge("s", f"l,{i}", weight=0, capacity=1)
        G.add_edge(f"r,{i}", "t", weight=0, capacity=1)

    for i, l in enumerate(required_allocs):
        for j, r in enumerate(required_allocs[i:], start=i):
            if not l.lvr.overlap(r.lvr):
                G.add_edge(
                    f"l,{i}", f"r,{j}", weight=max(0, r.size - l.size), capacity=1
                )

    flow_dict = nx.min_cost_flow(G)
    total_flow = 0
    allocs = {}
    reuses = {}

    for v, flow in flow_dict["s"].items():
        if v[0] == "r" and flow == 1:
            total_flow += 1
            allocs[v[2:]] = required_allocs_to_idx[int(v[2:])].size

    for u, vs in flow_dict.items():
        if u[0] != "l":
            continue
        for v, flow in vs.items():
            if v[0] != "r" or flow != 1:
                continue

            shared_object = u[2:]
            while shared_object not in allocs:
                shared_object = reuses[shared_object]
            reuses[v[2:]] = shared_object

            avail = allocs[shared_object]
            necessary = required_allocs_to_idx[int(v[2:])].size
            if avail < necessary:
                allocs[shared_object] = necessary

    planned_allocs = []
    planned_allocs_dict = {}
    offset = 0
    for alloc, size in allocs.items():
        req = required_allocs_to_idx[int(alloc)]
        lvr = LiveRange(req.lvr.begin, req.lvr.end)
        mem_region = MemRegion(offset, size)
        offset += size
        pl_alloc = PlannedAlloc(lvr, mem_region)
        planned_allocs.append(pl_alloc)
        planned_allocs_dict[alloc] = pl_alloc

    for tens, shared_obj in reuses.items():
        req = required_allocs_to_idx[int(tens)]
        lvr = LiveRange(req.lvr.begin, req.lvr.end)
        avail = planned_allocs_dict[shared_obj]
        assert not lvr.overlap(avail.lvr)
        assert req.size <= avail.mem_region.size, (req.size, avail.mem_region.size)

        reuse_start = planned_allocs_dict[shared_obj].mem_region.offset
        mem_region = MemRegion(reuse_start, req.size)
        planned_allocs.append(PlannedAlloc(lvr, mem_region))

    return planned_allocs


def solve_cp(required_allocs: List[RequiredAlloc]):
    model = cp_model.CpModel()

    max_size = sum(r.size for r in required_allocs)

    live_ranges = []
    offsets = []
    offsets_plus_sizes = []
    regions = []
    for i, r in enumerate(required_allocs):
        live_range = model.NewIntervalVar(
            r.lvr.begin, r.lvr.end + 1 - r.lvr.begin, r.lvr.end + 1, "live_range_%i" % i
        )
        live_ranges.append(live_range)

        offset = model.NewIntVar(0, max_size * 2, "offset_%i" % i)
        offset_plus_size = model.NewIntVar(0, max_size * 2, "offset_plus_size_%i" % i)
        region = model.NewIntervalVar(offset, r.size, offset_plus_size, "region_%i" % i)
        # model.Add(offset + size == offset_plus_size)

        offsets.append(offset)
        offsets_plus_sizes.append(offset_plus_size)
        regions.append(region)

    # Main constraint.
    model.AddNoOverlap2D(live_ranges, regions)

    total_size = model.NewIntVar(0, max_size * 2, "u")
    model.AddMaxEquality(total_size, offsets_plus_sizes)
    model.Minimize(total_size)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # # https://github.com/google/or-tools/blob/stable/ortools/sat/doc/model.md#model-copy
    # copy = cp_model.CpModel()
    # copy.CopyFrom(model)

    if status == cp_model.OPTIMAL:
        res = []
        for i, r in enumerate(required_allocs):
            res.append(PlannedAlloc.from_req(r, solver.Value(offsets[i])))
        return res
    else:
        print("no solution")
        return []


def solve_mip(required_allocs: List[RequiredAlloc]):
    solver = pywraplp.Solver.CreateSolver("SCIP")

    max_mem = sum(r.size for r in required_allocs)

    total_mem = solver.IntVar(0.0, max_mem, "u")
    offsets = []
    for i, r in enumerate(required_allocs):
        offset = solver.IntVar(0.0, max_mem, r.ptr_addr)
        # offset_i + mem_i <= total_mem
        solver.Add(offset + r.size <= total_mem)
        offsets.append(offset)

    # we encode the non-overlapping constraints using ordering of the blocks
    # if two blocks overlap then their allocations must be ordered
    # z_ij are decision variable. z_ij = 0 if block i has a lower
    # offset than block j i.e. offset_i + mem_i <= offset_j
    # and z_ij = 1 if the converse offset_j + mem_j <= offset_i
    # (note there could be a gap if we stick a block in between them
    for i, r1 in enumerate(required_allocs):
        for j, r2 in enumerate(required_allocs[i + 1 :], start=i + 1):
            if r1.lvr.overlap(r2.lvr):
                inters = solver.IntVar(0.0, 1, f"inters_{{{i},{j}}}")
                # if z_ij = 0 then i < j then offsets[i] + mems[i] <= offsets[j]
                # but if z_ij = 1 then j < i then offsets[i] + mems[i] <= offsets[j] + max_mem
                # otherwise the signs wouldn't be right
                solver.Add(
                    offsets[i] + required_allocs[i].size
                    <= offsets[j] + inters * max_mem
                )
                # conversely here
                solver.Add(
                    offsets[j] + required_allocs[j].size
                    <= offsets[i] + (1 - inters) * max_mem
                )

    # Minimize u
    solver.Minimize(total_mem)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return [
            PlannedAlloc.from_req(r, offsets[i].solution_value())
            for i, r in enumerate(required_allocs)
        ]
    else:
        print("The problem does not have an optimal solution.")
        return []


# def greedy_by_breadth(mem_events_per_op):
#     mem_events_per_op_ls = list(mem_events_per_op.items())
#     mem_events_per_op_ls.sort(key=lambda x: sum([y[1] for y in x[1]]), reverse=True)
#
#     mem_allocs = []
#     for _, mem_events in mem_events_per_op_ls:
#         mem_events.sort(key=lambda x: x[1], reverse=True)
#         mem_allocs.extend(mem_events)
#
#     ordered_allocs = []
#     inorder_of_decision_allocs = []
#     total_consumption = 0
#     for record in mem_allocs:
#         best_offset = find_gap(record, ordered_allocs)
#
#         (begin_t, end_t), size_t = record
#         total_consumption = max(total_consumption, best_offset + size_t)
#
#         inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
#         ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
#         ordered_allocs.sort(key=lambda x: x[1][0])
#
#     return inorder_of_decision_allocs, total_consumption


def calculate_high_watermark(allocs: List[PlannedAlloc]):
    allocs.sort(key=lambda p: p.lvr.begin)
    peak = 0
    for a in allocs:
        peak = max(peak, a.mem_region.next_free_addr)
    return int(peak)


def verify_allocation(allocations):
    for i, alloc1 in enumerate(allocations):
        for j, alloc2 in enumerate(allocations):
            if i == j:
                continue
            if alloc1.overlap(alloc2):
                print(alloc1, alloc2)
                return False
    return True


def lstm_test(lvrs):
    print("LSTMGreedyBySizeWithSmallestGap")
    res = greedy_by_size(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ],
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
    )
    res.sort(key=lambda x: x.lvr.begin)
    pprint(res)
    print()

    print("LSTMGreedyBySizeWithFirstGap")
    res = greedy_by_size(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ],
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST),
    )
    res.sort(key=lambda x: x.lvr.begin)
    pprint(res)
    print()

    print("LSTMGreedyByLongestAndSizeWithSmallestGap")
    res = greedy_by_longest(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ],
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
    )
    res.sort(key=lambda x: x.lvr.begin)
    pprint(res)
    print()

    print("LSTMGreedyByLongestAndSizeWithFirstGap")
    res = greedy_by_longest(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ],
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST),
    )
    res.sort(key=lambda x: x.lvr.begin)
    pprint(res)


if __name__ == "__main__":
    from profile_models import get_required_mem_allocs

    LSTM_lvrs = {
        (0, 3): 1024,
        (1, 3): 1024,
        (3, 4): 1024,
        (5, 10): 256,
        (6, 14): 256,
        (7, 10): 256,
        (8, 9): 256,
        (9, 12): 256,
        (10, 12): 256,
        (13, 14): 256,
    }

    req_mem_allocs = [
        RequiredAlloc(LiveRange(begin, end), size, str(i))
        for i, ((begin, end), size) in enumerate(LSTM_lvrs.items())
    ]

    trace_json = json.load(
        open(
            "/home/mlevental/dev_projects/pytorch_memory_planning/traces/resnet18,1x3x128x128.json"
        )
    )
    req_mem_allocs = get_required_mem_allocs(trace_json)

    res = gergov(req_mem_allocs)
    # res.sort(key=lambda r: r.lvr.begin)
    print(verify_allocation(res))
    print(calculate_high_watermark(res))
