#pragma once

#include "interval_tree.hpp"
#include "memory_planning.h"


std::vector<PlannedAlloc> greedyBySizeWithSmallestGap(
    const SortedLiveRangeMap<size_t>& live_ranges);

std::vector<PlannedAlloc> greedyBySizeWithFirstGap(
    const SortedLiveRangeMap<size_t>& live_ranges);

std::vector<PlannedAlloc> greedyByLongestAndSizeWithFirstGap(
    const SortedLiveRangeMap<size_t>& live_ranges);

std::vector<PlannedAlloc> greedyByLongestAndSizeWithSmallestGap(
    const SortedLiveRangeMap<size_t>& live_ranges);

std::vector<PlannedAlloc> gergov(
    const SortedLiveRangeMap<size_t>& live_ranges);

std::vector<PlannedAlloc> bump_allocator(
    const SortedLiveRangeMap<size_t>& live_ranges);

enum GAP_PRIORITY { FIRST, SMALLEST };

struct interval_hash {
  std::size_t operator()(const lib_interval_tree::interval_t<size_t>& p) const {
    auto h1 = std::hash<size_t>{}(p.low());
    auto h2 = std::hash<size_t>{}(p.high());

    return mem_hash::get_hash(h1, h2);
  }
};

struct tuple_hash {
  std::size_t operator()(const std::tuple<size_t, size_t, size_t>& p) const {
    auto h1 = std::hash<size_t>{}(std::get<0>(p));
    auto h2 = std::hash<size_t>{}(std::get<1>(p));
    auto h3 = std::hash<size_t>{}(std::get<2>(p));

    return mem_hash::get_hash(mem_hash::get_hash(h1, h2), h3);
  }
};