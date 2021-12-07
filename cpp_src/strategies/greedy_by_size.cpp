#include "greedy_by_size.h"
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>


#include "interval_tree.hpp"
using namespace lib_interval_tree;

bool lenCmp(
    std::pair<UniqueLiveRange, size_t> p1,
    std::pair<UniqueLiveRange, size_t> p2) {
  auto ulvr1 = p1.first;
  auto size1 = p1.second;
  auto ulvr2 = p2.first;
  auto size2 = p2.second;
  auto cmp = liveRangeStartCmp();

  auto len1 = ulvr1.lvr.end - ulvr1.lvr.begin;
  auto len2 = ulvr2.lvr.end - ulvr2.lvr.begin;
  return len1 == len2 ? (size1 == size2 ? cmp(ulvr1, ulvr2) : size1 > size2)
                      : len1 > len2;
}

// sort tensor usage records in non-increasing order of size (breaking ties by
// comparing live range starts)
bool sizeCmp(
    std::pair<UniqueLiveRange, size_t> p1,
    std::pair<UniqueLiveRange, size_t> p2) {
  auto ulvr1 = p1.first;
  auto size1 = p1.second;
  auto ulvr2 = p2.first;
  auto size2 = p2.second;
  auto cmp = liveRangeStartCmp();

  return size1 == size2 ? cmp(ulvr1, ulvr2) : size1 > size2;
}

using Cmp = bool(
    (std::pair<UniqueLiveRange, size_t> p1,
     std::pair<UniqueLiveRange, size_t> p2));

size_t findGapOffset(
    UniqueLiveRange unalloced_ulvr,
    size_t size,
    std::unordered_map<interval_t<size_t>, PlannedAlloc, interval_hash>
        current_allocations,
    interval_tree_t<size_t> current_allocation_tree,
    GAP_PRIORITY gap_priority) {
  size_t best_gap = std::numeric_limits<size_t>::max();
  int best_offset = -1;
  size_t prev_offset = 0;
//  std::cout << unalloced_ulvr << std::endl;

  std::vector<interval_t<size_t>> sorted_overlaps;
  auto iter = current_allocation_tree.overlap_find(
      {unalloced_ulvr.lvr.begin, unalloced_ulvr.lvr.end});
  for (; iter != current_allocation_tree.end(); iter++) {
    sorted_overlaps.emplace_back(iter->low(), iter->high());
  }
  std::sort(
      sorted_overlaps.begin(),
      sorted_overlaps.end(),
      [&current_allocations](auto intr1, auto intr2) {
        return current_allocations[intr1].reg.offset <
            current_allocations[intr2].reg.offset;
      });
  for (auto& item : sorted_overlaps) {
    auto alloc = current_allocations[item];
    if (alloc.reg.offset >= prev_offset) { // so that gap isn't negative...
      auto gap = alloc.reg.offset - prev_offset;
      if (size <= gap && gap < best_gap) {
        best_offset = prev_offset;
        if (gap_priority == GAP_PRIORITY::FIRST)
          break;
        best_gap = gap;
      }
    }
//    std::cout << "prev_offset " << prev_offset << std::endl;
    prev_offset = std::max(prev_offset, alloc.reg.nextOffset());
//    std::cout << "prev_offset " << prev_offset << std::endl;
//    std::cout << alloc << std::endl;
//    std::cout << "best_offset "
//              << best_offset
//              << std::endl;
//    std::cout << std::endl;
  }

  if (best_offset < 0) {
    best_offset = prev_offset;
  }
  return best_offset;
}

std::vector<PlannedAlloc> orderAllocations(
    std::unordered_map<interval_t<size_t>, PlannedAlloc, interval_hash>
        current_allocations) {
  std::vector<PlannedAlloc> ordered_allocations;
  ordered_allocations.reserve(current_allocations.size());
  for (auto& item : current_allocations) {
    ordered_allocations.emplace_back(item.second);
  }

  auto final_order_cmp = liveRangeStartCmp();
  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&final_order_cmp](auto m1, auto m2) {
        return final_order_cmp(m1.ulvr, m2.ulvr);
      });

  return ordered_allocations;
}

std::vector<PlannedAlloc> greedyBy(
    Cmp cmp,
    GAP_PRIORITY gap_priority,
    const SortedLiveRangeMap<size_t>& sorted_reqs) {
  std::vector<std::pair<UniqueLiveRange, size_t>> sorted_size_live_ranges(
      sorted_reqs.begin(), sorted_reqs.end());
  std::sort(
      sorted_size_live_ranges.begin(), sorted_size_live_ranges.end(), cmp);
  std::unordered_map<interval_t<size_t>, PlannedAlloc, interval_hash>
      current_allocations;
  //  for (const auto& item : sorted_size_live_ranges) {
  //    all_sizes.insert({{item.first.lvr.begin, item.first.lvr.end},
  //    item.second});
  //  }

  interval_tree_t<size_t> current_allocations_tree;
  for (auto& item : sorted_size_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    auto aligned_size = compute_aligned_tensor_size(size);
    auto offset = findGapOffset(
        ulvr,
        aligned_size,
        current_allocations,
        current_allocations_tree,
        gap_priority);
    current_allocations.insert(
        {{ulvr.lvr.begin, ulvr.lvr.end}, {{ulvr}, {offset, aligned_size}}});
    current_allocations_tree.insert({ulvr.lvr.begin, ulvr.lvr.end});
  }

  return orderAllocations(current_allocations);
}

std::vector<PlannedAlloc> greedyBySizeWithSmallestGap(
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(sizeCmp, GAP_PRIORITY::SMALLEST, live_ranges);
}

std::vector<PlannedAlloc> greedyBySizeWithFirstGap(
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(sizeCmp, GAP_PRIORITY::FIRST, live_ranges);
}

std::vector<PlannedAlloc> greedyByLongestAndSizeWithSmallestGap(
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(lenCmp, GAP_PRIORITY::SMALLEST, live_ranges);
}

std::vector<PlannedAlloc> greedyByLongestAndSizeWithFirstGap(
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(lenCmp, GAP_PRIORITY::FIRST, live_ranges);
}

std::vector<PlannedAlloc> bump_allocator(
    const SortedLiveRangeMap<size_t>& live_ranges) {
    std::vector<std::tuple<std::string, size_t, size_t>> mem_events;
    mem_events.reserve(2*live_ranges.size());

    for (auto& item: live_ranges) {
        auto ptr_addr = item.first.id;
        auto begin = item.first.lvr.begin;
        auto end = item.first.lvr.end;
        auto size = item.second;
        mem_events.emplace_back(std::make_tuple(ptr_addr, begin, size));
        mem_events.emplace_back(std::make_tuple(ptr_addr, end, -size));
    }

    std::sort(mem_events.begin(), mem_events.end(), [](auto m1, auto m2) {
        return std::get<0>(m1) < std::get<0>(m2);
    });

    std::unordered_map<std::string, PlannedAlloc> planned_allocs;

    int next_offset = 0;
    int curr_allocs = 0;

    std::vector<PlannedAlloc> real_planned_allocs;
    real_planned_allocs.reserve(planned_allocs.size());
    for (auto& item: mem_events) {
        auto ptr_addr = std::get<0>(item);
        auto ts = std::get<1>(item);
        auto size = std::get<2>(item);
        if (size > 0) {
            planned_allocs.insert({
                ptr_addr, {{{ts, 0}, ptr_addr}, {next_offset, size}}
            });
            next_offset += size;
            curr_allocs += 1;
        } else {
            planned_allocs[ptr_addr].ulvr.lvr.end = ts;
            real_planned_allocs.push_back(planned_allocs[ptr_addr]);
        }
        curr_allocs -= 1;
        if (curr_allocs == 0) {
            next_offset = 0;
        }
    }

  return real_planned_allocs;
}



typedef std::tuple<size_t, size_t, size_t> triple;

int first_available(std::vector<int> color_list) {
  std::unordered_set<int> color_set(color_list.begin(), color_list.end());
  auto count = 0;
  while (true) {
    if (!color_set.count(count)) {
      return count;
    }
    count++;
  }
}

std::unordered_map<std::string, int> greedy_color(
    std::unordered_map<std::string, std::vector<std::string>> E,
    std::vector<std::string> order) {
  std::unordered_map<std::string, int> color;
  for (auto& u : order) {
    std::vector<int> used_neighbor_colors;
    for (auto& v : E[u]) {
      if (color.count(v)) {
        used_neighbor_colors.push_back(color[v]);
      }
    }
    color[u] = first_available(used_neighbor_colors);
  }
  return color;
}

std::vector<PlannedAlloc> gergov(
    const SortedLiveRangeMap<size_t>& live_ranges) {
  auto l = std::min_element(
               live_ranges.begin(),
               live_ranges.end(),
               [](auto item1, auto item2) {
                 return item1.first.lvr.begin < item2.first.lvr.begin;
               })
               ->first.lvr.begin;
  auto r = std::max_element(
               live_ranges.begin(),
               live_ranges.end(),
               [](auto item1, auto item2) {
                 return item1.first.lvr.end < item2.first.lvr.end;
               })
               ->first.lvr.end;

  std::unordered_map<interval_t<size_t>, size_t, interval_hash> H;
  interval_tree_t<size_t> H_tree;

  auto add_to_H = [&H, &H_tree](size_t ll, size_t rr, size_t ww) {
    H.insert({{ll, rr}, ww});
    H_tree.insert({ll, rr});
  };
  auto remove_from_H = [&H, &H_tree](size_t rrr, size_t ccc) {
    if (H.count({rrr, ccc}))
      H.erase({rrr, ccc});
    auto iter = H_tree.find({rrr, ccc});
    if (iter != H_tree.end())
      H_tree.erase(iter);
  };

  add_to_H(l, r, 0);

  std::unordered_map<interval_t<size_t>, size_t, interval_hash> J;
  interval_tree_t<size_t> J_tree;
  auto add_to_J = [&J, &J_tree](size_t rrrr, size_t cccc, size_t ssss) {
    J.insert({{rrrr, cccc}, ssss});
    J_tree.insert({rrrr, cccc});
  };
  auto remove_from_J = [&J, &J_tree](size_t rrrrr, size_t ccccc) {
    if (J.count({rrrrr, ccccc}))
      J.erase({rrrrr, ccccc});
    auto iter = J_tree.find({rrrrr, ccccc});
    if (iter != J_tree.end())
      J_tree.erase(iter);
  };

  std::unordered_map<interval_t<size_t>, size_t, interval_hash> intr_to_size;
  std::unordered_map<interval_t<size_t>, std::string, interval_hash> intr_to_id;
  std::unordered_map<std::string, UniqueLiveRange> id_to_ulvr;
  for (auto& item : live_ranges) {
    add_to_J(item.first.lvr.begin, item.first.lvr.end, item.second);
    intr_to_size.insert(
        {{item.first.lvr.begin, item.first.lvr.end}, item.second});
    intr_to_id.insert(
        {{item.first.lvr.begin, item.first.lvr.end}, item.first.id});
    id_to_ulvr.insert({item.first.id, item.first});
  }

  auto if_there_exists = [&J_tree, &H_tree](interval_t<size_t> intr) {
    interval_t<size_t> row(-1, -1);
    auto iter = J_tree.overlap_find(intr);

    while (iter != J_tree.end()) {
      if (H_tree.overlap_find(iter->interval()) == H_tree.end()) {
        return iter->interval();
      }
      iter++;
    }

    return row;
  };

  std::vector<interval_t<size_t>> V;
  interval_tree_t<size_t> VV_tree;
  std::unordered_map<triple, size_t, tuple_hash> alphap;
  std::vector<std::string> order;
  std::unordered_map<std::string, triple> id_to_src;

  while (!J.empty()) {
    auto pick_idx =
        std::min_element(H.begin(), H.end(), [](auto itr1, auto itr2) {
          return itr1.second < itr2.second;
        });
    if (pick_idx == H.end())
      break;

    auto w = pick_idx->second;
    auto xl = pick_idx->first.low();
    auto xr = pick_idx->first.high();
    auto intr = pick_idx->first;
    remove_from_H(xl, xr);

    auto j = if_there_exists(intr);
    if (j.low() == -1 || j.high() == -1)
      continue;

    auto s = intr_to_size[j];
    auto r = j.low();
    auto c = j.high();
    remove_from_J(r, c);

    V.push_back(j);
    order.push_back(intr_to_id[j]);
    triple t(s, r, c);
    id_to_src[intr_to_id[j]] = t;
    alphap[t] = w;
    VV_tree.insert(j);

    auto ma = std::max(xl, r);
    auto mi = std::min(c, xr);

    add_to_H(ma, mi, w + s);
    if (xl < r) {
      add_to_H(xl, r, w);
    }

    if (c < xr) {
      add_to_H(c, xr, w);
    }
  }

  interval_tree_t<size_t> alphapp_tree;
  for (auto& item : V) {
    auto s = intr_to_size[item];
    auto r = item.low();
    auto c = item.high();
    auto ap_vi = alphap[{s, r, c}];
    alphapp_tree.insert({ap_vi, ap_vi + s});
  }
  std::unordered_map<std::string, std::vector<std::string>> E;
  for (auto& i : V) {
    auto si = intr_to_size[i];
    auto ri = i.low();
    auto ci = i.high();
    auto ap_vi = alphap[{si, ri, ci}];
    interval<size_t> vi{ap_vi, ap_vi + si};
    auto j = VV_tree.overlap_find(i);
    for (; j != VV_tree.end() && i != *j; j++) {
      auto sj = intr_to_size[*j];
      auto rj = j->low();
      auto cj = j->high();
      auto ap_vj = alphap[{sj, rj, cj}];
      interval<size_t> vj{ap_vj, ap_vj + sj};
      if (vi.overlaps(vj)) {
        auto u = intr_to_id[i];
        auto v = intr_to_id[*j];
        E[u].push_back(v);
        E[v].push_back(u);
      }
    }
  }

  auto color = greedy_color(E, order);

  size_t maxj = 0;
  for (auto& item : V) {
    auto s = intr_to_size[item];
    auto r = item.low();
    auto c = item.high();
    triple t(s, r, c);
    maxj = std::max(maxj, alphap[t] + s);
  }
  std::unordered_map<triple, size_t, tuple_hash> alpha;
  for (auto& item : color) {
    auto id = item.first;
    auto trip = id_to_src[id];
    alpha[trip] = alphap[trip] + item.second * maxj;
  }

  std::vector<PlannedAlloc> planned_allocs;
  for (auto& item : alpha) {
    auto s = std::get<0>(item.first);
    auto r = std::get<1>(item.first);
    auto c = std::get<2>(item.first);
    PlannedAlloc m({{r, c}, {item.second, s}});
    planned_allocs.push_back(m);
  }

  return planned_allocs;
}

