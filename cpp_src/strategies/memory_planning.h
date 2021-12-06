#pragma once

#include <iostream>
#include <map>
#include <vector>
#include "hash.h"

constexpr size_t gAlignment = 64;

enum class Strategy {
  NAIVE = 0,
  GREEDY_BY_SIZE_WITH_SMALLEST_GAP,
  GREEDY_BY_SIZE_WITH_FIRST_GAP,
  GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP,
  GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP,
  GERGOV
};

inline size_t compute_aligned_tensor_size(size_t nbytes) {
  // Note: everything below is size_t
  return (nbytes + gAlignment - 1) & (~(gAlignment - 1));
}

inline const char* toString(Strategy s) {
  switch (s) {
    case Strategy::NAIVE:
      return "NAIVE";
    case Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP:
      return "GREEDY_BY_SIZE_WITH_SMALLEST_GAP";
    case Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP:
      return "GREEDY_BY_SIZE_WITH_FIRST_GAP";
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP:
      return "GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP";
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP:
      return "GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP";
    case Strategy::GERGOV:
      return "GERGOV";
    default:
      return "UNKNOWN STRATEGY";
  }
}

inline std::ostream& operator<<(std::ostream& str, Strategy rhs) {
  return str << toString(rhs);
}

typedef struct MemRegion {
  size_t offset;
  size_t size;

  size_t nextOffset() const {
    return offset + size;
  }

} MemRegion;

inline std::ostream& operator<<(std::ostream& str, MemRegion reg) {
  return str << "{offset: " << reg.offset << ", size: " << reg.size << "}";
}

inline bool operator==(const MemRegion& lhs, const MemRegion& rhs) {
  return lhs.offset == rhs.offset && lhs.size == rhs.size;
}

struct regionSizeCmp {
  bool operator()(const MemRegion& reg1, const MemRegion& reg2) const {
    return reg1.size == reg2.size ? reg1.offset < reg2.offset
                                  : reg1.size < reg2.size;
  }
};

struct regionOffsetCmp {
  bool operator()(const MemRegion& reg1, const MemRegion& reg2) const {
    return reg1.offset == reg2.offset ? reg1.size < reg2.size
                                      : reg1.offset < reg2.offset;
  }
};

bool overlapMemRegion(const MemRegion& reg1, const MemRegion& reg2);

typedef struct LiveRange {
  size_t begin;
  size_t end;
} LiveRange;

inline std::ostream& operator<<(std::ostream& str, LiveRange lvr) {
  return str << "[" << lvr.begin << ", " << lvr.end << "]";
}

inline bool operator==(const LiveRange& lhs, const LiveRange& rhs) {
  return lhs.begin == rhs.begin && lhs.end == rhs.end;
}

struct UniqueLiveRange {
  LiveRange lvr;
  std::string id;
};

bool overlapLiveRange(
    const UniqueLiveRange& ulvr1,
    const UniqueLiveRange& ulvr2);

inline std::ostream& operator<<(std::ostream& str, UniqueLiveRange rhs) {
  return str << "{id: " << rhs.id << ", lvr: " << rhs.lvr << "}";
}

inline bool operator==(const UniqueLiveRange lhs, const UniqueLiveRange rhs) {
  return lhs.lvr == rhs.lvr && lhs.id == rhs.id;
}

struct liveRangeStartCmp {
  bool operator()(const UniqueLiveRange& u1, const UniqueLiveRange& u2) const {
    return u1.lvr.begin == u2.lvr.begin
        ? (u1.lvr.end == u2.lvr.end ? u1.id < u2.id : u1.lvr.end < u2.lvr.end)
        : u1.lvr.begin < u2.lvr.begin;
  }
};

struct liveRangeEndCmp {
  bool operator()(const UniqueLiveRange& u1, const UniqueLiveRange& u2) const {
    return u1.lvr.end == u2.lvr.end
        ? (u1.lvr.begin == u2.lvr.begin ? u1.id < u2.id
                                        : u1.lvr.begin < u2.lvr.begin)
        : u1.lvr.end < u2.lvr.end;
  }
};

template <typename T>
using SortedLiveRangeMap = std::map<UniqueLiveRange, T, liveRangeStartCmp>;

struct MemAllocation {
  UniqueLiveRange ulvr;
  MemRegion reg;
};

struct memAllocOffsetCmp {
  regionOffsetCmp cmp;
  bool operator()(const MemAllocation& u1, const MemAllocation& u2) const {
    return cmp(u1.reg, u2.reg);
  }
};

inline std::ostream& operator<<(std::ostream& str, MemAllocation rhs) {
  return str << rhs.ulvr << ", " << rhs.reg;
}

void printAllocations(std::vector<MemAllocation> allocations);

inline bool operator==(const MemAllocation lhs, const MemAllocation rhs) {
  return lhs.ulvr == rhs.ulvr && lhs.reg == rhs.reg;
}

inline bool valid_add(size_t a, size_t b) {
#if defined(_MSC_VER)
  return a + b >= a;
#else
  size_t _carry = 0;
  return !__builtin_add_overflow(a, b, &_carry);
#endif
}

inline bool valid_sub(size_t a, size_t b) {
#if defined(_MSC_VER)
  return a >= b;
#else
  size_t _carry = 0;
  return !__builtin_sub_overflow(a, b, &_carry);
#endif
}

// size, allocations
using MemoryPlan = std::pair<size_t, std::vector<MemAllocation>>;

MemoryPlan planMemory(
    SortedLiveRangeMap<size_t> managed_live_ranges,
    Strategy strat);

namespace _hash_detail {

// Use template argument deduction to shorten calls to c10::hash
template <typename T>
size_t simple_get_hash(const T& o);

template <typename T, typename V>
using type_if_not_enum =
    typename std::enable_if<!std::is_enum<T>::value, V>::type;

// Use SFINAE to dispatch to std::hash if possible, cast enum types to int
// automatically, and fall back to T::hash otherwise. NOTE: C++14 added support
// for hashing enum types to the standard, and some compilers implement it even
// when C++14 flags aren't specified. This is why we have to disable this
// overload if T is an enum type (and use the one below in this case).
template <typename T>
auto dispatch_hash(const T& o)
    -> decltype(std::hash<T>()(o), type_if_not_enum<T, size_t>()) {
  return std::hash<T>()(o);
}

template <typename T>
typename std::enable_if<std::is_enum<T>::value, size_t>::type dispatch_hash(
    const T& o) {
  using R = typename std::underlying_type<T>::type;
  return std::hash<R>()(static_cast<R>(o));
}

template <typename T>
auto dispatch_hash(const T& o) -> decltype(T::hash(o), size_t()) {
  return T::hash(o);
}

} // namespace _hash_detail

// Hasher struct
template <typename T>
struct hash {
  size_t operator()(const T& o) const {
    return _hash_detail::dispatch_hash(o);
  };
};


namespace std {

template <>
struct hash<MemRegion> {
  size_t operator()(const MemRegion& reg) const {
    return mem_hash::get_hash(reg.offset, reg.size);
  }
};

template <>
struct hash<UniqueLiveRange> {
  size_t operator()(const UniqueLiveRange& ulvr) const {
    return mem_hash::get_hash(ulvr.lvr, ulvr.id);
  }
};

template <>
struct hash<MemAllocation> {
  size_t operator()(const MemAllocation& mem) const {
    return mem_hash::get_hash(mem.reg, mem.ulvr);
  }
};

template <>
struct hash<LiveRange> {
  size_t operator()(LiveRange const& range) const {
    // shift so that single point ranges don't have hash zero (xor cancels)
    return mem_hash::get_hash(range.begin, range.end);
  }
};

}


