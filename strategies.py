from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import partial, total_ordering
from pprint import pprint
from typing import List, Dict

from intervaltree import IntervalTree
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from z3 import Optimize, Bools, Not, And


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
        return overlap(self.offset, self.size, other.begin, other.end)

    @property
    def next_free_addr(self):
        return self.offset + self.size

    def __str__(self):
        return f"{(self.offset, self.size)}"


@dataclass
class RequiredAlloc:
    lvr: LiveRange
    size: int
    ptr_addr: str

    def __str__(self):
        return f"{self.lvr}:{self.size}"


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
        interval_tree: IntervalTree,
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
    # biggest size first but break ties deterministically
    req_mem_allocs.sort(key=lambda r: (r.size, r.lvr), reverse=True)
    return _greedy_by_size(req_mem_allocs, gap_finder)


def greedy_by_longest(
        req_mem_allocs: List[RequiredAlloc],
        *,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
):
    req_mem_allocs.sort(key=lambda r: (len(r.lvr), r.lvr), reverse=True)
    return _greedy_by_size(req_mem_allocs, gap_finder)


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


def solve_cp(required_allocs: List[RequiredAlloc]):
    model = cp_model.CpModel()

    max_size = sum(r.size for r in required_allocs)

    live_ranges = []
    offsets = []
    offsets_plus_sizes = []
    regions = []
    for i, r in enumerate(required_allocs):
        live_range = model.NewIntervalVar(
            r.lvr.begin, r.lvr.end - r.lvr.begin, r.lvr.end, "live_range_%i" % i
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
        for j, r2 in enumerate(required_allocs[i + 1:], start=i + 1):
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


def verify_allocation(allocations):
    for i, alloc1 in enumerate(allocations):
        for j, alloc2 in enumerate(allocations):
            if i == j:
                continue
            if alloc1.overlap(alloc2):
                return False
    return True


if __name__ == "__main__":
    lvrs = {
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

# TODO: buddy allocator
# TODO: mincost flow
# TODO: graph coloring
