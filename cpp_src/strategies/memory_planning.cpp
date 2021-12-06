#include "memory_planning.h"
#include "greedy_by_size.h"

#include <algorithm>
#include <limits>


// overlap of [a, b) and [c, d)
bool overlap(size_t a, size_t b, size_t c, size_t d) {
  assert(a <= b);
  assert(c <= d);
  size_t outer_len = std::max(b, d) - std::min(a, c);
  size_t interval_len_1 = (b - a), interval_len_2 = (d - c);

  // overflow checking since we're dealing with size_t arithmetic
  // as of today (09/02/2021) linear address width on x86 is 48bits (256TB)
  // so this is unneccessary but "640kB [isn't] enough for anyone"
  // so this is necessary
  if (!valid_add(interval_len_1, interval_len_2) ||
      !valid_sub(outer_len, interval_len_1 + interval_len_2)) {
    // multipoint overlap (sum areas larger than outer area)
    return true;
  } else {
    return false;
  }
}

// live ranges overlap like closed intervals (i.e. if .end of one is the same as
// .begin of another)
// note that we add 1 to the right endpoint since [a,b] doesn't intersect [c,d]
// if [a,b+1) doesn't intersect [c,d+1) (since all endpoints are integers).
bool overlapLiveRange(
    const UniqueLiveRange& ulvr1,
    const UniqueLiveRange& ulvr2) {
  return overlap(
      ulvr1.lvr.begin, ulvr1.lvr.end + 1, ulvr2.lvr.begin, ulvr2.lvr.end + 1);
}

// since memory address are zero indexed, offset + size ends at the *beginning*
// of the (offset+size)th byte. hence overlap is like open intervals (i.e.
// overlap is more than at endpoints)
bool overlapMemRegion(const MemRegion& reg1, const MemRegion& reg2) {
  return overlap(
      reg1.offset,
      reg1.offset + reg1.size,
      reg2.offset,
      reg2.offset + reg2.size);
}

bool overlapAllocs(const MemAllocation& m1, const MemAllocation& m2) {
  return overlapLiveRange(m1.ulvr, m2.ulvr) && overlapMemRegion(m1.reg, m2.reg);
}

// stack all tensors end to end "as you see them" over the entire lifespan of
// the plan
std::vector<MemAllocation> naive(
    SortedLiveRangeMap<size_t> managed_live_ranges) {
  std::vector<MemAllocation> allocations;
  allocations.reserve(managed_live_ranges.size());
  size_t offset = 0;
  for (const auto& item : managed_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    auto id = item.first.id;
    auto aligned_size = compute_aligned_tensor_size(size);
    allocations.push_back({ulvr, {offset, aligned_size}});
    offset += aligned_size;
  }
  return allocations;
}

void printAllocations(std::vector<MemAllocation> allocations) {
  std::cout << "\"allocations\": {\n";
  for (const auto& item : allocations) {
    std::cout << "(" << item.ulvr.lvr.begin << "," << item.ulvr.lvr.end << ")"
              << ":"
              << "(" << item.reg.offset << "," << item.reg.size << "),\n";
  }
  std::cout << "},\n"
            << "\n";
}

void printLiveRangesAndSizes(
    std::vector<std::pair<UniqueLiveRange, size_t>> lvrs) {
  auto cmp = liveRangeStartCmp();
  std::sort(lvrs.begin(), lvrs.end(), [&cmp](auto a, auto b) {
    return cmp(a.first, b.first);
  });
  std::cout << "\"live_ranges_sizes\": {\n";
  for (const auto& item : lvrs) {
    std::cout << "(" << item.first.lvr.begin << "," << item.first.lvr.end << ")"
              << ":" << item.second << ",\n";
  }
  std::cout << "},\n"
            << "\n";
}

// "high watermark" of memory
size_t getTotalAllocationSize(std::vector<MemAllocation> allocations) {
  size_t total_size = 0;
  for (const auto& alloc : allocations) {
    total_size = std::max(total_size, alloc.reg.offset + alloc.reg.size);
  }
  return total_size;
}

bool validateAllocations(
    std::vector<MemAllocation> allocations,
    SortedLiveRangeMap<size_t> managed_live_ranges,
    size_t total_size) {
  if (total_size >= (size_t)std::numeric_limits<int64_t>::max()) {
    //    GRAPH_DEBUG("total allocation is too big ", total_size);
    return false;
  }

  for (const auto& alloc1 : allocations) {
    for (const auto& alloc2 : allocations) {
      if (alloc1 == alloc2) {
        continue;
      }
      if (overlapAllocs(alloc1, alloc2)) {
        std::cerr << "overlapping allocations: " << alloc1 << ", " << alloc2
                  << std::endl;
        return false;
      }
    }
  }

  if (allocations.size() != managed_live_ranges.size()) {
    std::cerr << "not the right number of allocations: " << allocations.size()
              << ";" << managed_live_ranges.size() << std::endl;
    return false;
  }

  for (const auto& alloc : allocations) {
    if (!valid_add(alloc.reg.offset, alloc.reg.size)) {
      std::cerr << "allocation " << alloc.reg
                << " beyond int64_t mem limit: " << sizeof(int64_t)
                << std::endl;
      return false;
    }

    if (!valid_sub(total_size, alloc.reg.offset + alloc.reg.size)) {
      // if this overflows then alloc.reg.offset + alloc.reg.size > total_size
      std::cerr << "allocation exceeds total size: " << alloc.reg << ", "
                << total_size << std::endl;
      return false;
    }

    if (managed_live_ranges.count(alloc.ulvr) == 0 ||
        // leq because word alignment increases requirements
        // (recomputing aligned size is overkill here)
        managed_live_ranges[alloc.ulvr] > alloc.reg.size) {
      std::cerr << "wrong size allocation: " << alloc.ulvr << ", "
                << managed_live_ranges[alloc.ulvr] << ", " << alloc.reg.size
                << std::endl;
      return false;
    }
  }

  return true;
}

MemoryPlan planMemory(
    SortedLiveRangeMap<size_t> managed_live_ranges,
    Strategy strat) {

  std::vector<MemAllocation> allocations;
  switch (strat) {
    case Strategy::NAIVE: {
      allocations = naive(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP: {
      allocations = greedyBySizeWithSmallestGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP: {
      allocations = greedyBySizeWithFirstGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP: {
      allocations = greedyByLongestAndSizeWithSmallestGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP: {
      allocations = greedyByLongestAndSizeWithFirstGap(managed_live_ranges);
      break;
    }
    case Strategy::GERGOV: {
      allocations = gergov(managed_live_ranges);
      break;
    }
    default:
      return {};
  }

  auto total_size = getTotalAllocationSize(allocations);

  return std::make_pair(total_size, allocations);
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(memory_planning, m) {
    py::class_<MemRegion>(m, "MemRegion")
        .def(py::init<const size_t &, const size_t &>())
        .def_readwrite("offset", &MemRegion::offset)
        .def_readwrite("size", &MemRegion::size);

    py::class_<LiveRange>(m, "LiveRange")
        .def(py::init<const size_t &, const size_t &>())
        .def_readwrite("offset", &LiveRange::begin)
        .def_readwrite("size", &LiveRange::end);

    py::class_<UniqueLiveRange>(m, "UniqueLiveRange")
        .def(py::init<const LiveRange &, const std::string &>())
        .def_readwrite("offset", &UniqueLiveRange::lvr)
        .def_readwrite("size", &UniqueLiveRange::id);

    py::class_<MemAllocation>(m, "MemAllocation")
//        .def(py::init<LiveRange, MemRegion>())
        .def_readwrite("ulvr", &MemAllocation::ulvr)
        .def_readwrite("reg", &MemAllocation::reg);

    py::enum_<Strategy>(m, "Strategy")
        .value("NAIVE", Strategy::NAIVE)
        .value("GREEDY_BY_SIZE_WITH_SMALLEST_GAP", Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP)
        .value("GREEDY_BY_SIZE_WITH_FIRST_GAP", Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP)
        .value("GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP", Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP)
        .value("GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP", Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP)
        .value("GERGOV", Strategy::GERGOV)
        .export_values();

    m.def("planMemory", &planMemory, "plan memory");
}

