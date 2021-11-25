#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream> // for ostringstream
#include <string>
#include <jemalloc/jemalloc.h>
#include <pybind11/pybind11.h>

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

void write_cb(void *opaque, const char *to_write) {
  auto *arg = (std::ofstream *)opaque;
  size_t bytes = strlen(to_write);
  arg->write(to_write, bytes);
}

void je_malloc_print_stats(const std::string &json_fp) {
  try {
    std::ofstream json_stream;
    json_stream.open(json_fp);
    jemalloc_stats_print(write_cb, (void *)&json_stream, "");
  } catch (const std::exception& e) {
    std::cout << e.what() << "\n";
  }
}

void do_something(size_t i) { jemalloc(i * 100); }
void test_jemalloc() {
  for (size_t i = 0; i < 1000; i++) {
    do_something(i);
  }

  je_malloc_print_stats("test_je_malloc.json");
}


static uint64_t epoch = 1;
std::string get_stats() {
  // Update the statistics cached by mallctl.
  epoch += 1;
  size_t sz;
  sz = sizeof(epoch);
  jemallctl("epoch", &epoch, &sz, &epoch, sz);

  // Get basic allocation statistics.  Take care to check for
  // errors, since --enable-stats must have been specified at
  // build time for these statistics to be available.
  size_t allocated, active, metadata, resident, mapped, retained;
  std::ostringstream out;
  sz = sizeof(size_t);
  if (jemallctl("stats.allocated", &allocated, &sz, NULL, 0) == 0 &&
      jemallctl("stats.active", &active, &sz, NULL, 0) == 0 &&
      jemallctl("stats.metadata", &metadata, &sz, NULL, 0) == 0 &&
      jemallctl("stats.resident", &resident, &sz, NULL, 0) == 0 &&
      jemallctl("stats.mapped", &mapped, &sz, NULL, 0) == 0 &&
      jemallctl("stats.retained", &retained, &sz, NULL, 0) == 0) {
//    sprintf(buffer,
//            "Current allocated/active/metadata/resident/mapped/retained: "
//            "%zu/%zu/%zu/%zu/%zu/%zu\n",
//            allocated, active, metadata, resident, mapped, retained);
    out << allocated << "," << active << "," << metadata << "," << resident << "," << mapped << "," << retained;
  }
  return out.str();
}

void reset_je_malloc_stats() {
  jemallctl("stats.mutexes.reset", NULL, NULL, NULL, 0);
}

void set_num_background_threads(size_t n) {
  size_t sz;
  sz = sizeof(n);
  jemallctl("max_background_threads", &n, &sz, &n, sz);
}

void dump_je_heap_profile(const std::string &prof_fp) {
  auto prof_c_str = prof_fp.c_str();
//  jemallctl("prof.dump", NULL, NULL, &prof_c_str, sizeof(const char *));
  jemallctl("prof.dump", NULL, NULL, NULL, 0);
}

namespace py = pybind11;

// this module name needs to be the same as the name in cmake
PYBIND11_MODULE(jemalloc_bindings, m) {
  m.def("test_jemalloc", &test_jemalloc);
  m.def("je_malloc", &jemalloc);
  m.def("je_free", &jefree);
  m.def("je_malloc_print_stats", &je_malloc_print_stats);
  m.def("reset_je_malloc_stats", &reset_je_malloc_stats);
  m.def("dump_je_heap_profile", &dump_je_heap_profile);
  m.def("get_stats", &get_stats);
  m.def("set_num_background_threads", &set_num_background_threads);
}
