from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import partial, total_ordering
from operator import attrgetter
from typing import List, Dict

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


def intersect(xs, ys):
    return max(xs[1], ys[1]) - min(xs[0], ys[0]) - (xs[1] - xs[0]) - (ys[1] - ys[0])


def intersect_mem(xs, ys):
    return intersect(xs, ys) < 0


def intersect_lvr(xs, ys):
    return intersect(xs, ys) <= 0


def intersect_allocs(alloc1, alloc2):
    ((begin1, end1), (offset1, size1)) = alloc1
    ((begin2, end2), (offset2, size2)) = alloc2
    interse = intersect_lvr((begin1, end1), (begin2, end2)) and intersect_mem(
        (offset1, offset1 + size1), (offset2, offset2 + size2)
    )

    if interse:
        print((begin1, end1), (begin2, end2))
        print((offset1, offset1 + size1), (offset2, offset2 + size2))

    return interse


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

    def intersect(self, other):
        return intersect_lvr((self.begin, self.end), (other.begin, other.end))


@dataclass
class RequiredAlloc:
    lvr: LiveRange
    size: int
    ptr_addr: str


@dataclass
class PlannedAlloc:
    lvr: LiveRange
    offset: int
    size: int

    @property
    def next_free_addr(self):
        return self.offset + self.size

    @classmethod
    def from_req(cls, req_alloc: RequiredAlloc, offset: int):
        return PlannedAlloc(req_alloc.lvr, offset, req_alloc.size)


class GapPriority(Enum):
    SMALLEST = 1
    FIRST = 2


def find_gap(
    record: RequiredAlloc,
    ordered_allocs: List[PlannedAlloc],
    *,
    GAP_PRIORITY: GapPriority = GapPriority.SMALLEST,
):
    best_gap = float("inf")
    best_offset = None
    prev_offset = 0

    for alloc in ordered_allocs:
        if not intersect_lvr(
            (alloc.lvr.begin, alloc.lvr.end), (record.lvr.begin, record.lvr.end)
        ):
            continue

        # offset_x will be ahead of the previous block
        # while prev_offset will be just in front
        # this looks for small gap ahead of a block
        gap = alloc.offset - prev_offset
        if record.size <= gap < best_gap:
            best_offset = prev_offset
            if GAP_PRIORITY == GapPriority.FIRST:
                break
            best_gap = gap

        prev_offset = max(prev_offset, alloc.next_free_addr)

    if best_offset is None:
        best_offset = prev_offset
    return best_offset


def _greedy_by_size(
    sorted_req_mem_allocs: List[RequiredAlloc],
    gap_finder,
):
    ordered_allocs: List[PlannedAlloc] = []
    inorder_of_decision_allocs: List[PlannedAlloc] = []
    for mem_alloc in sorted_req_mem_allocs:
        best_offset = gap_finder(mem_alloc, ordered_allocs)

        p = PlannedAlloc.from_req(mem_alloc, best_offset)
        inorder_of_decision_allocs.append(p)
        ordered_allocs.append(p)

        # sort by offset
        ordered_allocs.sort(key=attrgetter("offset"))

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
    req_mem_allocs.sort(key=lambda r: len(r.lvr), reverse=True)
    return _greedy_by_size(req_mem_allocs, gap_finder)


MemEvent = namedtuple("MemEvent", "ptr_addr size ts")


def bump_allocator(mem_events: List[MemEvent]):
    mem_events.sort(key=lambda r: r.ts)
    planned_allocations: Dict[str, PlannedAlloc] = {}
    next_offset = 0
    curr_allocs = 0
    for ptr_addr, size, ts in mem_events:
        if size > 0:
            planned_allocations[ptr_addr] = PlannedAlloc(
                LiveRange(ts, -1), next_offset, size
            )
            next_offset += size
            curr_allocs += 1
        elif size < 0:
            assert ptr_addr in planned_allocations
            assert planned_allocations[ptr_addr].size == -size
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
            if r1.lvr.intersect(r2.lvr):
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
            if intersect_allocs(alloc1, alloc2):
                return False
    return True
